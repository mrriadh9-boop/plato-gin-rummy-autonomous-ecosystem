"""
Match Lifecycle State Machine & Auto-Navigation Manager for Plato Gin Rummy.
Handles seamless transitions between Matchmaking Lobby, In-Game Table,
Round Summary, Match Victory/Defeat Summary, and Modal Popups.
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from driver.dispatcher.touch_dispatcher import ADBTapDispatcher
from vision.ocr.text_detector import FastTextOCR

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    """Current high-level lifecycle state in Plato Gin Rummy."""
    UNKNOWN = "UNKNOWN"
    LOBBY = "LOBBY"                  # Main menu / Ranked match selection
    MATCHMAKING = "MATCHMAKING"      # Waiting for opponent queue
    INGAME = "INGAME"                # Active Gin Rummy table match
    ROUND_SUMMARY = "ROUND_SUMMARY"  # Intermediate score tally between hands
    MATCH_SUMMARY = "MATCH_SUMMARY"  # Final victory / defeat screen (100 pts)
    MODAL_POPUP = "MODAL_POPUP"      # Generic dialog / confirmation popup
    ERROR_DISCONNECTED = "ERROR_DISCONNECTED"  # Disconnection banner


@dataclass
class LifecycleDetection:
    """Detection result from inspecting the raw game screen."""
    state: LifecycleState
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_tap: Optional[Tuple[int, int]] = None
    reason: str = ""


@dataclass
class MatchStats:
    """Session-level aggregate statistics across matches."""
    matches_played: int = 0
    matches_won: int = 0
    total_rounds_played: int = 0
    total_gin_calls: int = 0
    total_undercuts: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def win_rate(self) -> float:
        return (self.matches_won / self.matches_played * 100.0) if self.matches_played > 0 else 0.0


class MatchLifecycleManager:
    """
    Autonomous state machine that manages transitions across Plato Gin Rummy.
    """

    def __init__(self, device_serial: Optional[str] = "2ace61d0", ocr_reader: Optional[FastTextOCR] = None):
        self.device_serial = device_serial
        self.ocr = ocr_reader or FastTextOCR()
        self.dispatcher = ADBTapDispatcher(device_serial=device_serial)
        self.current_state = LifecycleState.UNKNOWN
        self.state_history: List[Tuple[float, LifecycleState]] = []
        self.last_transition_time = time.time()
        self.stats = MatchStats()

        # Plato Ranked Match queue coordinates on 1800x2880 Xiaomi Pad 6 canvas
        self.RANKED_MATCH_BTN = (1679, 993)
        self.CONFIRM_JOIN_BTN = (1218, 1592)
        self.MATCH_EXIT_BTN = (80, 80)
        self.PLAY_AGAIN_BTN = (933, 1659)

    def detect_lifecycle_state(self, frame: np.ndarray) -> LifecycleDetection:
        """
        Inspects the raw frame to accurately classify the lifecycle state.
        """
        if frame is None or frame.size == 0:
            return LifecycleDetection(state=LifecycleState.UNKNOWN, confidence=0.0, reason="Empty frame")

        h, w = frame.shape[:2]

        # 1. Match Summary / Victory / Defeat Screen Check (Priority 1)
        mid_strip = frame[int(0.35 * h):int(0.68 * h), int(0.10 * w):int(0.90 * w)]
        raw_text = ""
        if self.ocr.reader:
            try:
                txt_list = self.ocr.reader.readtext(mid_strip, detail=0)
                raw_text = " ".join(txt_list)
            except Exception:
                pass

        if re.search(r"\b(won|forfeited|lost|victory|defeat|ranking|vs|rematch|play again)\b", raw_text, re.IGNORECASE):
            # Target exact PLAY AGAIN button at (933, 1659) on 1800x2880 canvas
            return LifecycleDetection(
                state=LifecycleState.MATCH_SUMMARY,
                confidence=0.98,
                details={"text": raw_text},
                recommended_tap=(933, 1659),
                reason=f"Match Summary detected: '{raw_text}'"
            )

        # 2. Lobby / Ranked Match Selection Check (Priority 2)
        lobby_strip = frame[int(0.10 * h):int(0.45 * h), int(0.05 * w):int(0.95 * w)]
        lobby_text = ""
        if self.ocr.reader:
            try:
                lobby_txt_list = self.ocr.reader.readtext(lobby_strip, detail=0)
                lobby_text = " ".join(lobby_txt_list)
            except Exception:
                pass

        if re.search(r"\b(ranked|gin rummy|join|create|custom|leaderboard|play)\b", lobby_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.LOBBY,
                confidence=0.95,
                details={"text": lobby_text},
                recommended_tap=self.RANKED_MATCH_BTN,
                reason=f"Lobby Menu detected: '{lobby_text}'"
            )

        # 3. Intermediate Round Summary Check (Score Tally between hands)
        if re.search(r"\b(round|deadwood|bonus|score|next hand|continue)\b", raw_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.ROUND_SUMMARY,
                confidence=0.95,
                details={"text": raw_text},
                recommended_tap=(900, 2500),
                reason=f"Round Summary detected: '{raw_text}'"
            )

        # 4. In-Game Green Felt Detection (Color Check)
        felt_roi = frame[int(0.35 * h):int(0.55 * h), int(0.20 * w):int(0.80 * w)]
        hsv_felt = cv2.cvtColor(felt_roi, cv2.COLOR_BGR2HSV)
        # Deep green table felt mask: H: 35-85, S: 40-255, V: 20-200
        lower_green = np.array([35, 30, 20])
        upper_green = np.array([85, 255, 220])
        mask = cv2.inRange(hsv_felt, lower_green, upper_green)
        green_ratio = float(np.count_nonzero(mask) / mask.size)

        if green_ratio > 0.40:
            return LifecycleDetection(
                state=LifecycleState.INGAME,
                confidence=0.95,
                details={"green_ratio": green_ratio},
                reason=f"Table felt detected (Green ratio: {green_ratio:.2f})"
            )

        # 5. Modal Popup Confirmation
        if re.search(r"\b(confirm|ok|cancel|accept|ready|start)\b", raw_text, re.IGNORECASE):
            return LifecycleDetection(
                state=LifecycleState.MODAL_POPUP,
                confidence=0.85,
                details={"text": raw_text},
                recommended_tap=self.CONFIRM_JOIN_BTN,
                reason=f"Modal popup detected: '{raw_text}'"
            )

        # Fallback to UNKNOWN
        return LifecycleDetection(
            state=LifecycleState.UNKNOWN,
            confidence=0.20,
            details={"raw_text": raw_text, "green_ratio": green_ratio},
            reason="Unrecognized screen layout"
        )

    def execute_lifecycle_action(self, detection: LifecycleDetection) -> bool:
        """
        Executes the necessary navigation tap or ADB action to keep the bot in active ranked play.
        """
        now = time.time()
        # Debounce transitions by at least 1.5s
        if now - self.last_transition_time < 1.5:
            return False

        state = detection.state

        if state == LifecycleState.MATCH_SUMMARY:
            logger.info(f"[Lifecycle] Match Summary Detected -> Tapping PLAY AGAIN at {detection.recommended_tap or (933, 1659)}")
            tap_coords = detection.recommended_tap or (933, 1659)
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
            logger.info(f"[Lifecycle] Lobby Detected -> Tapping JOIN Ranked Match at (1679, 993) + Confirm (1218, 1592)")
            self.dispatcher.tap(1679, 993)
            time.sleep(0.6)
            self.dispatcher.tap(1218, 1592)
            self.last_transition_time = now
            return True

        elif state == LifecycleState.MODAL_POPUP:
            logger.info(f"[Lifecycle] Modal / Confirmation Popup Detected -> Tapping Confirm at (1218, 1592)")
            self.dispatcher.tap(1218, 1592)
            self.last_transition_time = now
            return True

        return False
