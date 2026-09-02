"""
Match Lifecycle Manager for Plato Gin Rummy Autonomous Ranked Auto-Play.
Detects and automates all match transitions: Lobby -> Queue -> In-Game -> Round End -> Match End -> Auto-Requeue.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from driver.dispatcher.touch_dispatcher import ADBTapDispatcher
from vision.pipeline.scoreboard_ocr import ScoreboardOCR

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    """Lifecycle phases in Plato Gin Rummy."""
    LOBBY = "LOBBY"                      # In Plato game selector / lobby
    MATCHMAKING = "MATCHMAKING"          # Finding ranked opponent spinner
    INGAME = "INGAME"                    # Active card game table
    ROUND_SUMMARY = "ROUND_SUMMARY"      # Hand finished, deadwood points tallying
    MATCH_SUMMARY = "MATCH_SUMMARY"      # Full match finished (>=100 pts), winner announced
    MODAL_POPUP = "MODAL_POPUP"          # Daily reward / disconnect / notification dialog
    UNKNOWN = "UNKNOWN"                  # Unrecognized screen


@dataclass
class MatchStats:
    """Historical telemetry for autonomous ranked sessions."""
    matches_played: int = 0
    matches_won: int = 0
    matches_lost: int = 0
    total_rounds_played: int = 0
    rating_points_delta: int = 0
    current_rating: int = 0
    last_match_score: Tuple[int, int] = (0, 0)
    session_start_time: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return (self.matches_won / self.matches_played) * 100.0


@dataclass
class LifecycleDetection:
    """Detection output for current screen frame."""
    state: LifecycleState
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_tap: Optional[Tuple[int, int]] = None
    reason: str = ""


class MatchLifecycleManager:
    """
    Orchestrates out-of-game transitions so the bot can play 24/7 ranked matches
    without human input.
    """

    def __init__(self, ocr_reader: Optional[ScoreboardOCR] = None, dispatcher: Optional[ADBTapDispatcher] = None):
        self.ocr = ocr_reader or ScoreboardOCR()
        self.dispatcher = dispatcher or ADBTapDispatcher()
        self.stats = MatchStats()

        # Debounce timestamps
        self.last_transition_time = 0.0
        self.last_state = LifecycleState.UNKNOWN
        self.state_stable_count = 0
        self.current_match_id = f"match_{int(time.time())}"
        self._in_match_recorded = False

    def detect_lifecycle_state(self, frame: np.ndarray) -> LifecycleDetection:
        """
        Classifies screen frame into one of the LifecycleState phases.
        Hardware resolution: 1800x2880.
        """
        if frame is None or frame.size == 0:
            return LifecycleDetection(state=LifecycleState.UNKNOWN, confidence=0.0, reason="Empty frame")

        h, w = frame.shape[:2]

        # 1. Quick In-Game Table Check
        # Felt color profile in center
        center_crop = frame[int(0.30 * h):int(0.60 * h), int(0.20 * w):int(0.80 * w)]
        avg_bgr = np.mean(center_crop, axis=(0, 1))
        is_felt_background = bool((avg_bgr[1] > avg_bgr[0] + 8) and (avg_bgr[1] > avg_bgr[2] + 8))

        # Check turn indicators / buttons
        is_turn = self.ocr.is_player_turn(frame)
        is_knock = self.ocr.is_knock_available(frame)
        is_pass = self.ocr.is_pass_available(frame)

        # Bottom HUD deadwood check
        dw_val = self.ocr.read_deadwood(frame)

        if is_turn or is_knock or is_pass or dw_val is not None:
            return LifecycleDetection(
                state=LifecycleState.INGAME,
                confidence=0.99,
                details={"is_turn": is_turn, "knock": is_knock, "pass": is_pass, "deadwood": dw_val},
                reason="In-game table controls active"
            )

        # 2. Match Summary / Victory Screen Check
        mid_strip = frame[int(0.40 * h):int(0.65 * h), int(0.10 * w):int(0.90 * w)]
        raw_text = ""
        if self.ocr.reader:
            try:
                txt_list = self.ocr.reader.readtext(mid_strip, detail=0)
                raw_text = " ".join(txt_list)
            except Exception:
                pass

        if re.search(r"\b(won|victory|defeat|ranking|vs)\b", raw_text, re.IGNORECASE):
            play_again_coords = (900, 2550)
            return LifecycleDetection(
                state=LifecycleState.MATCH_SUMMARY,
                confidence=0.95,
                details={"text": raw_text},
                recommended_tap=play_again_coords,
                reason=f"Match Summary detected: '{raw_text}'"
            )

        # 3. Round Summary Check
        if re.search(r"\b(gin|knock|undercut|bonus|round)\b", raw_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.ROUND_SUMMARY,
                confidence=0.90,
                details={"text": raw_text},
                recommended_tap=(900, 2500),
                reason="Round score summary active"
            )

        # 4. Lobby / Ranked Queue Selection Check
        bottom_nav = frame[int(0.80 * h):int(0.98 * h), int(0.10 * w):int(0.90 * w)]
        nav_text = ""
        if self.ocr.reader:
            try:
                nav_list = self.ocr.reader.readtext(bottom_nav, detail=0)
                nav_text = " ".join(nav_list)
            except Exception:
                pass

        if re.search(r"\b(ranked|play|find match|leaderboard|start)\b", nav_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.LOBBY,
                confidence=0.88,
                details={"nav_text": nav_text},
                recommended_tap=(900, 2500),
                reason="Lobby / Play button detected"
            )

        # 5. Modal / Dialog Popup Check
        if re.search(r"\b(ok|claim|cancel|reconnect|close)\b", raw_text + " " + nav_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.MODAL_POPUP,
                confidence=0.85,
                details={"text": raw_text},
                recommended_tap=(900, 1600),
                reason="Modal popup detected"
            )

        # Default fallback: If felt background detected, assume in-game waiting
        if is_felt_background:
            return LifecycleDetection(
                state=LifecycleState.INGAME,
                confidence=0.75,
                details={"felt": True},
                reason="Table felt present, waiting for opponent/dealer"
            )

        return LifecycleDetection(
            state=LifecycleState.UNKNOWN,
            confidence=0.30,
            details={"raw_text": raw_text},
            reason="Unclassified screen state"
        )

    def execute_lifecycle_action(self, detection: LifecycleDetection, cooldown_s: float = 2.0) -> bool:
        """
        Executes touch event to advance through non-gameplay screens autonomously.
        Returns True if an out-of-game action was dispatched.
        """
        now = time.perf_counter()
        if now - self.last_transition_time < cooldown_s:
            return False

        state = detection.state

        if state == LifecycleState.MATCH_SUMMARY:
            logger.info(f"[Lifecycle] Match Summary Detected -> Dispatching Play Again / Next Match tap")
            tap_coords = detection.recommended_tap or (900, 2550)
            self.dispatcher.tap(tap_coords[0], tap_coords[1])
            self.last_transition_time = now
            self.stats.matches_played += 1
            return True

        elif state == LifecycleState.ROUND_SUMMARY:
            logger.info(f"[Lifecycle] Round Summary Detected -> Tapping to advance to next hand")
            tap_coords = detection.recommended_tap or (900, 2500)
            self.dispatcher.tap(tap_coords[0], tap_coords[1])
            self.last_transition_time = now
            self.stats.total_rounds_played += 1
            return True

        elif state == LifecycleState.LOBBY:
            logger.info(f"[Lifecycle] Lobby Detected -> Tapping Ranked Match / Play button")
            tap_coords = detection.recommended_tap or (900, 2500)
            self.dispatcher.tap(tap_coords[0], tap_coords[1])
            self.last_transition_time = now
            return True

        elif state == LifecycleState.MODAL_POPUP:
            logger.info(f"[Lifecycle] Modal Popup Detected -> Dismissing modal")
            tap_coords = detection.recommended_tap or (900, 1600)
            self.dispatcher.tap(tap_coords[0], tap_coords[1])
            self.last_transition_time = now
            return True

        return False
