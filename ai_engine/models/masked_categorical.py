"""
MaskedCategorical Action Distribution for Gin Rummy.
Provides exact mathematically sound logit masking for illegal moves with device and dtype robustness.
"""

from typing import Optional, Union
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class MaskedCategorical(Categorical):
    MASK_VALUE: float = -1e8

    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: Optional[Union[torch.Tensor, torch.BoolTensor]] = None,
        validate_args: Optional[bool] = None,
    ):
        if logits is not None and mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool, device=logits.device)
            else:
                mask = mask.to(device=logits.device, dtype=torch.bool)

            if mask.shape != logits.shape:
                mask = mask.expand_as(logits)

            all_masked = (~mask).all(dim=-1, keepdim=True)
            effective_mask = torch.where(all_masked, torch.ones_like(mask), mask)

            mask_val = -1e8 if logits.dtype == torch.float32 else torch.finfo(logits.dtype).min
            masked_logits = torch.where(
                effective_mask,
                logits,
                torch.full_like(logits, mask_val),
            )
            self._mask = effective_mask
            super().__init__(logits=masked_logits, validate_args=validate_args)
        elif probs is not None and mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool, device=probs.device)
            else:
                mask = mask.to(device=probs.device, dtype=torch.bool)

            if mask.shape != probs.shape:
                mask = mask.expand_as(probs)

            all_masked = (~mask).all(dim=-1, keepdim=True)
            effective_mask = torch.where(all_masked, torch.ones_like(mask), mask)

            masked_probs = torch.where(
                effective_mask,
                probs,
                torch.zeros_like(probs),
            )
            prob_sum = masked_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            normalized_probs = masked_probs / prob_sum
            self._mask = effective_mask
            super().__init__(probs=normalized_probs, validate_args=validate_args)
        else:
            self._mask = mask
            super().__init__(logits=logits, probs=probs, validate_args=validate_args)

    @property
    def mask(self) -> Optional[torch.Tensor]:
        return self._mask

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.logits, dim=-1)

    def entropy(self) -> torch.Tensor:
        log_p = F.log_softmax(self.logits, dim=-1)
        p = self.probs
        p_log_p = torch.where(p > 1e-12, p * log_p, torch.zeros_like(p))
        return -p_log_p.sum(dim=-1)


CategoricalMasked = MaskedCategorical
