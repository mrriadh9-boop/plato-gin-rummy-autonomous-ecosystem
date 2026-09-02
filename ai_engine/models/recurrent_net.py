"""
Recurrent Neural Policy & Value Network for Plato Gin Rummy.

Implements RecurrentGinRummyNet:
- 8 Spatial Card Planes: (B, 8, 4, 13)
  * Plane 0: Player Hand
  * Plane 1: Top Discard
  * Plane 2: Opponent Known Pickups
  * Plane 3: Opponent Discards
  * Plane 4: Player Discards
  * Plane 5: Passed Discards
  * Plane 6: Unknown Pool (Stock + Hidden Opponent)
  * Plane 7: Melded / Deadwood Safety Matrix
- 16 Context Scalars: (B, 16)
  * Cumulative match scores towards 100 threshold, score diff, stock count,
    turn number, deadwood, discard-lock state, opponent draw telemetry.
- Dual-stream 2D convolutions (1x3/1x13 runs, 4x1 sets).
- Dense trunk with LayerNorm and SiLU activations.
- 256-dim GRU temporal memory cell.
- 110-action MaskedCategorical actor head (illegal actions set to -1e8).
- Scalar Value head V(s).
- 52-dim Opponent Belief head (predicting opponent's 52-card holdings).
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_engine.models.masked_categorical import MaskedCategorical


class RecurrentGinRummyNet(nn.Module):
    """
    Recurrent Neural Policy & Value Network for Plato Gin Rummy.
    """

    def __init__(
        self,
        num_actions: int = 110,
        in_channels: int = 8,
        num_scalars: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.num_scalars = num_scalars
        self.hidden_dim = hidden_dim

        # 1. Rank convolution branch (detects runs along ranks for each suit)
        self.rank_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 2. Suit convolution branch (detects sets across suits for each rank)
        self.suit_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(4, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 3. Global feature dense branch (fuses all flat cards + context scalars)
        total_flat_dim = in_channels * 52 + num_scalars  # 8*52 + 16 = 432
        self.global_fc = nn.Sequential(
            nn.Linear(total_flat_dim, 256),
            nn.ReLU(),
        )

        # 4. Dense feature trunk
        combined_dim = (64 * 4 * 13) + (32 * 1 * 13) + 256  # 3328 + 416 + 256 = 4000
        self.trunk = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # 5. Recurrent GRU memory layer
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 6. Prediction Heads
        self.actor_head = nn.Linear(hidden_dim, num_actions)
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.belief_head = nn.Linear(hidden_dim, 52)

        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal weight initialization with proper gains."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.GRUCell):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

        # Smaller initialization for policy actor head and belief head
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0.0)
        nn.init.orthogonal_(self.belief_head.weight, gain=0.5)
        nn.init.constant_(self.belief_head.bias, 0.0)

    def _process_inputs(
        self,
        obs: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observation and scalars into standard 2D grid (B, 8, 4, 13) and flat (B, 432).
        Handles automatic padding for legacy 5-plane / 6-plane inputs.
        """
        batch_size = obs.shape[0]
        device = obs.device

        if obs.dim() == 4 and obs.shape[1] == self.in_channels:
            # (Batch, 8, 4, 13)
            obs_2d = obs
            obs_flat_cards = obs.reshape(batch_size, self.in_channels * 52)
        elif obs.dim() == 3 and obs.shape[1] == self.in_channels and obs.shape[2] == 52:
            # (Batch, 8, 52)
            obs_2d = obs.reshape(batch_size, self.in_channels, 4, 13)
            obs_flat_cards = obs.reshape(batch_size, self.in_channels * 52)
        elif obs.dim() == 3 and obs.shape[1] < self.in_channels and obs.shape[2] == 52:
            # (Batch, C < 8, 52) -> Pad to 8 planes
            pad_count = self.in_channels - obs.shape[1]
            pad_plane = torch.zeros(batch_size, pad_count, 52, device=device, dtype=obs.dtype)
            obs_padded = torch.cat([obs, pad_plane], dim=1)
            obs_2d = obs_padded.reshape(batch_size, self.in_channels, 4, 13)
            obs_flat_cards = obs_padded.reshape(batch_size, self.in_channels * 52)
        elif obs.dim() == 4 and obs.shape[1] < self.in_channels and obs.shape[2] == 4 and obs.shape[3] == 13:
            # (Batch, C < 8, 4, 13) -> Pad to 8 planes
            pad_count = self.in_channels - obs.shape[1]
            pad_plane = torch.zeros(batch_size, pad_count, 4, 13, device=device, dtype=obs.dtype)
            obs_2d = torch.cat([obs, pad_plane], dim=1)
            obs_flat_cards = obs_2d.reshape(batch_size, self.in_channels * 52)
        elif obs.dim() == 2:
            total_dim = obs.shape[1]
            expected_total = self.in_channels * 52 + self.num_scalars  # 432
            if total_dim == expected_total:
                cards_part = obs[:, : self.in_channels * 52]
                scalars = obs[:, self.in_channels * 52 :]
                obs_2d = cards_part.reshape(batch_size, self.in_channels, 4, 13)
                obs_flat_cards = cards_part
            elif total_dim == (self.in_channels * 52):  # 416
                obs_2d = obs.reshape(batch_size, self.in_channels, 4, 13)
                obs_flat_cards = obs
            elif total_dim in (260, 312, 324):  # Legacy 5 or 6 planes
                if total_dim == 324:
                    cards_6p = obs[:, :312]
                    leg_scalars = obs[:, 312:]
                elif total_dim == 312:
                    cards_6p = obs
                    leg_scalars = None
                else:  # 260
                    cards_6p = torch.cat([obs, torch.zeros(batch_size, 52, device=device, dtype=obs.dtype)], dim=-1)
                    leg_scalars = None

                pad_plane = torch.zeros(batch_size, (self.in_channels - 6) * 52, device=device, dtype=obs.dtype)
                cards_8p = torch.cat([cards_6p, pad_plane], dim=-1)
                obs_2d = cards_8p.reshape(batch_size, self.in_channels, 4, 13)
                obs_flat_cards = cards_8p

                if scalars is None and leg_scalars is not None:
                    pad_sc = torch.zeros(batch_size, self.num_scalars - leg_scalars.shape[1], device=device, dtype=obs.dtype)
                    scalars = torch.cat([leg_scalars, pad_sc], dim=-1)
            else:
                raise ValueError(f"Unsupported 2D observation shape: {obs.shape}")
        else:
            raise ValueError(f"Unsupported observation shape: {obs.shape}")

        if scalars is None:
            scalars = torch.zeros(batch_size, self.num_scalars, device=device, dtype=obs.dtype)
        elif scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        if scalars.shape[1] < self.num_scalars:
            pad_sc = torch.zeros(batch_size, self.num_scalars - scalars.shape[1], device=device, dtype=scalars.dtype)
            scalars = torch.cat([scalars, pad_sc], dim=-1)

        total_flat = torch.cat([obs_flat_cards, scalars], dim=-1)
        return obs_2d, total_flat

    def _forward_trunk(
        self,
        obs: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract spatial features and step GRU layer."""
        obs_2d, total_flat = self._process_inputs(obs, scalars)
        batch_size = obs_2d.shape[0]

        rank_feat = self.rank_conv(obs_2d)
        suit_feat = self.suit_conv(obs_2d)
        global_feat = self.global_fc(total_flat)

        combined = torch.cat([rank_feat, suit_feat, global_feat], dim=-1)
        trunk_out = self.trunk(combined)

        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_dim, device=obs.device, dtype=trunk_out.dtype)
        elif hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        next_hidden = self.gru(trunk_out, hidden_state)
        return next_hidden, next_hidden

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[MaskedCategorical, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning masked action distribution, critic value, recurrent state, and belief logits.

        Returns:
            dist: MaskedCategorical action distribution (110 actions)
            values: Estimated state value V(s) of shape (Batch, 1)
            next_hidden: Recurrent hidden state of shape (Batch, 256)
            belief_logits: Opponent hand prediction logits of shape (Batch, 52)
        """
        recurrent_feat, next_hidden = self._forward_trunk(obs, scalars, hidden_state)
        logits = self.actor_head(recurrent_feat)
        values = self.critic_head(recurrent_feat)
        belief_logits = self.belief_head(recurrent_feat)
        dist = MaskedCategorical(logits=logits, mask=action_mask)
        return dist, values, next_hidden, belief_logits

    def get_value(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Estimate state value V(s)."""
        recurrent_feat, _ = self._forward_trunk(obs, scalars, hidden_state)
        return self.critic_head(recurrent_feat)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper method for PPO/TRPO rollout collection and policy updates.

        Returns:
            action: Chosen action (Batch,)
            log_prob: Action log-probability (Batch,)
            entropy: Policy entropy (Batch,)
            value: State value V(s) (Batch, 1)
            next_hidden: Recurrent hidden state (Batch, 256)
            belief_logits: Opponent card prediction logits (Batch, 52)
        """
        dist, value, next_hidden, belief_logits = self.forward(
            obs=obs,
            action_mask=action_mask,
            hidden_state=hidden_state,
            scalars=scalars,
        )
        if action is None:
            action = dist.mode() if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, next_hidden, belief_logits

    def predict_opponent_hand(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return estimated 52-card opponent hand probability distribution (Batch, 52)."""
        recurrent_feat, _ = self._forward_trunk(obs, scalars, hidden_state)
        belief_logits = self.belief_head(recurrent_feat)
        return torch.sigmoid(belief_logits)


GinRummyNet = RecurrentGinRummyNet
