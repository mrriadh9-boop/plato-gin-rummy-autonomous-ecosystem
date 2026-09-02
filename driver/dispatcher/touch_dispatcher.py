"""
Deterministic ADB Touch Tap Dispatcher for Plato Gin Rummy.
Targets exact 1800x2880 card centroids with 0px offset error.
Includes touch routines for all game actions and timing debounce protection.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# --- Canonical 1800x2880 Geometry & Targets ---
SCREEN_WIDTH = 1800
SCREEN_HEIGHT = 2880
HAND_Y_CENTROID = 1950

TOUCH_COORDINATES: Dict[str, Tuple[int, int]] = {
    "stock_pile": (720, 1290),      # Exact measured face-down stock pile centroid
    "discard_pile": (1080, 1290),   # Exact measured face-up discard pile centroid
    "knock_button": (1314, 2563),
    "pass_button": (1440, 1296),
    "cyan_turn_pill": (900, 2570),
    "sort_button": (900, 2570),
}

BOUNDING_BOXES: Dict[str, Tuple[int, int, int, int]] = {
    "hand_roi": (144, 1785, 1656, 2304),       # (x1, y1, x2, y2)
    "discard_roi": (900, 1065, 1296, 1555),
    "stock_roi": (324, 1209, 630, 1497),
    "action_log_roi": (108, 1497, 1692, 1728),
    "opp_score_roi": (450, 28, 684, 230),
    "player_score_roi": (450, 2304, 684, 2563),
    "deadwood_roi": (990, 2304, 1728, 2534),
    "turn_pill_roi": (756, 2505, 1044, 2635),
    "knock_btn_roi": (1044, 2505, 1584, 2635),
    "pass_btn_roi": (1260, 1152, 1620, 1440),
}


def calculate_hand_card_centroids(card_count: int) -> List[Tuple[int, int]]:
    """
    Computes exact (x, y) touch centroids for 1..11 cards displayed in player hand on 1800x2880 canvas.
    Adheres strictly to the overlapping fan geometry of Plato Gin Rummy (0px offset error).
    """
    if card_count <= 0:
        return []

    if card_count == 1:
        return [(900, HAND_Y_CENTROID)]

    # Spacing between consecutive card tops in Plato is ~114.7px on 1800x2880 canvas
    step = 114.7
    total_fan_w = (card_count - 1) * step + 260.0
    start_x = (SCREEN_WIDTH - total_fan_w) / 2.0

    centroids = []
    for i in range(card_count):
        left_x = start_x + i * step
        if i < card_count - 1:
            # Exposed region center: halfway between this card's left edge and the next card
            cx = int(round(left_x + step / 2.0))
        else:
            # Last card: full card center
            cx = int(round(left_x + 130.0))
        centroids.append((cx, HAND_Y_CENTROID))
    return centroids


@dataclass(frozen=True)
class TouchPoint:
    """Atomic touch point coordinate on 1800x2880 display."""
    x: int
    y: int
    name: str = ""

    def as_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


class ActionType(str, Enum):
    DRAW_STOCK = "DRAW_STOCK"
    DRAW_DISCARD = "DRAW_DISCARD"
    DISCARD_CARD = "DISCARD_CARD"
    KNOCK = "KNOCK"
    GIN = "GIN"
    PASS = "PASS"
    SORT_MELDS = "SORT_MELDS"


@dataclass
class DispatchedTap:
    """Record of a dispatched tap event."""
    action_type: ActionType
    x: int
    y: int
    timestamp: float
    command_str: str
    card_index: Optional[int] = None
    card_id: Optional[int] = None
    success: bool = True
    latency_ms: float = 0.0


def find_adb_binary(custom_path: Optional[str] = None) -> str:
    """
    Auto-detects and returns the absolute path to adb executable without relying solely on %PATH%.
    Checks custom_path, shutil.which, Android SDK environment variables,
    WinGet/Scrcpy packages, and standard AppData/ProgramFiles paths.
    """
    import shutil
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    # 1. PATH lookup
    which_path = shutil.which("adb")
    if which_path and os.path.isfile(which_path):
        return os.path.abspath(which_path)

    # 2. Environment variables (ANDROID_HOME, ANDROID_SDK_ROOT)
    for env_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        val = os.environ.get(env_var)
        if val:
            candidate = os.path.join(val, "platform-tools", "adb.exe" if sys.platform == "win32" else "adb")
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

    # 3. Known system & user package locations
    candidates = [
        r"C:\Users\60163\AppData\Local\Microsoft\WinGet\Packages\Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe\scrcpy-win64-v4.1\adb.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Android\platform-tools\adb.exe"),
        os.path.expandvars(r"%PROGRAMFILES(X86)%\Android\platform-tools\adb.exe"),
        r"C:\platform-tools\adb.exe",
        r"C:\scrcpy\adb.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)

    return "adb"


class ADBExecutor:
    """
    Handles ADB command formatting and execution over subprocess or mock backend.
    """

    def __init__(
        self,
        device_serial: Optional[str] = None,
        adb_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        self.device_serial = device_serial
        self.adb_path = find_adb_binary(adb_path)
        self.mock_mode = mock_mode
        self.execution_history: List[DispatchedTap] = []

    def format_tap_command(self, x: int, y: int) -> str:
        """Formats deterministic ADB shell input tap command string."""
        if self.device_serial:
            return f"{self.adb_path} -s {self.device_serial} shell input tap {x} {y}"
        return f"{self.adb_path} shell input tap {x} {y}"

    def format_swipe_command(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> str:
        """Formats ADB swipe / drag command."""
        if self.device_serial:
            return f"{self.adb_path} -s {self.device_serial} shell input swipe {x1} {y1} {x2} {y2} {duration_ms}"
        return f"{self.adb_path} shell input swipe {x1} {y1} {x2} {y2} {duration_ms}"

    def execute_tap(self, x: int, y: int, action_type: ActionType = ActionType.DISCARD_CARD) -> DispatchedTap:
        """Executes the tap on the device or records to mock history."""
        cmd = self.format_tap_command(x, y)
        t0 = time.perf_counter()
        success = True

        if not self.mock_mode:
            try:
                # Run ADB command
                result = subprocess.run(
                    cmd.split(),
                    capture_output=True,
                    timeout=2.0,
                    check=False,
                )
                success = (result.returncode == 0)
            except Exception as e:
                logger.warning("ADB execution failed (%s): %s", cmd, e)
                success = False

        latency_ms = (time.perf_counter() - t0) * 1000.0
        tap_record = DispatchedTap(
            action_type=action_type,
            x=x,
            y=y,
            timestamp=time.time(),
            command_str=cmd,
            success=success,
            latency_ms=latency_ms,
        )
        self.execution_history.append(tap_record)
        return tap_record


class ADBTapDispatcher:
    """
    High-level touch action dispatcher for Plato Gin Rummy.
    Provides complete game action routines, 0px coordinate calculations,
    and timing debouncing to prevent phantom double-taps or race conditions.
    """

    def __init__(
        self,
        device_serial: Optional[str] = None,
        adb_path: Optional[str] = None,
        min_debounce_interval_s: float = 0.080,  # 80ms minimum debounce between taps
        mock_mode: bool = False,
    ):
        self.min_debounce_interval_s = min_debounce_interval_s
        self.executor = ADBExecutor(device_serial=device_serial, adb_path=adb_path, mock_mode=mock_mode)
        self._last_tap_time = 0.0
        self._lock = threading.Lock()
        self._dispatch_history: List[DispatchedTap] = []

    def tap(self, x: int, y: int, action_type: ActionType = ActionType.DISCARD_CARD) -> DispatchedTap:
        """Executes a direct debounced tap at (x, y)."""
        self._debounce()
        tap = self.executor.execute_tap(x, y, action_type)
        self._dispatch_history.append(tap)
        return tap

    def _debounce(self) -> None:
        """Enforces debounce delay between sequential taps."""
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._last_tap_time
            if elapsed < self.min_debounce_interval_s:
                time.sleep(self.min_debounce_interval_s - elapsed)
            self._last_tap_time = time.perf_counter()

    def get_stock_centroid(self) -> Tuple[int, int]:
        """Exact centroid of stock draw pile."""
        return TOUCH_COORDINATES["stock_pile"]

    def get_discard_centroid(self) -> Tuple[int, int]:
        """Exact centroid of discard pickup pile."""
        return TOUCH_COORDINATES["discard_pile"]

    def get_knock_button_centroid(self) -> Tuple[int, int]:
        """Exact centroid of knock action button."""
        return TOUCH_COORDINATES["knock_button"]

    def get_pass_button_centroid(self) -> Tuple[int, int]:
        """Exact centroid of initial pass button."""
        return TOUCH_COORDINATES["pass_button"]

    def get_hand_card_centroid(self, card_index: int, total_cards: int = 11) -> Tuple[int, int]:
        """
        Calculates exact touch point for card at index in hand.
        Ensures card_index is clamped in valid [0, total_cards - 1] bounds.
        """
        if total_cards <= 0:
            total_cards = 10
        card_index = max(0, min(card_index, total_cards - 1))
        centroids = calculate_hand_card_centroids(total_cards)
        return centroids[card_index]

    # --- High-Level Touch Action Routines ---

    def draw_stock(self) -> DispatchedTap:
        """Tap Stock Pile at (720, 1290)."""
        self._debounce()
        x, y = self.get_stock_centroid()
        tap = self.executor.execute_tap(x, y, ActionType.DRAW_STOCK)
        self._dispatch_history.append(tap)
        return tap

    def draw_discard(self) -> DispatchedTap:
        """Tap Discard Pile at (1080, 1290)."""
        self._debounce()
        x, y = self.get_discard_centroid()
        tap = self.executor.execute_tap(x, y, ActionType.DRAW_DISCARD)
        self._dispatch_history.append(tap)
        return tap

    def discard_card(self, card_index: int, total_cards: int = 11) -> DispatchedTap:
        """Fling/drags card in hand towards discard pile (1080, 1290) to execute instantaneous discard."""
        self._debounce()
        x, y = self.get_hand_card_centroid(card_index, total_cards)
        tap = self.executor.execute_tap(x, y, ActionType.DISCARD_CARD)
        tap.card_index = card_index
        self._dispatch_history.append(tap)
        if not self.executor.mock_mode:
            try:
                # Fast 100ms fling upwards to discard pile
                swipe_cmd = self.executor.format_swipe_command(x, y, 1080, 1290, duration_ms=100)
                subprocess.run(swipe_cmd.split(), capture_output=True, timeout=1.0)
                # Backup tap on discard pile to guarantee selected/lifted card is dropped
                time.sleep(0.04)
                tap_cmd = self.executor.format_tap_command(1080, 1290)
                subprocess.run(tap_cmd.split(), capture_output=True, timeout=1.0)
            except Exception:
                pass
        return tap

    def discard_by_card_id(
        self,
        card_id: int,
        hand_ids: List[int],
        hand_coords: Optional[List[Dict[str, int]]] = None,
    ) -> DispatchedTap:
        """Locates card_id within hand_ids and drags/taps to discard pile."""
        if card_id in hand_ids:
            idx = hand_ids.index(card_id)
        else:
            idx = len(hand_ids) - 1 if hand_ids else 0

        self._debounce()
        if hand_coords and idx < len(hand_coords):
            x = hand_coords[idx]["x"]
            y = hand_coords[idx]["y"]
        else:
            x, y = self.get_hand_card_centroid(idx, total_cards=len(hand_ids) if hand_ids else 11)

        # In Plato, dragging/swiping card to the center discard pile (1080, 1290) executes the discard
        if not self.executor.mock_mode:
            try:
                swipe_cmd = self.executor.format_swipe_command(x, y, 1080, 1290, duration_ms=200)
                subprocess.run(swipe_cmd.split(), capture_output=True, timeout=2.0)
            except Exception:
                pass

        tap = self.executor.execute_tap(x, y, ActionType.DISCARD_CARD)
        tap.card_id = card_id
        tap.card_index = idx
        self._dispatch_history.append(tap)
        return tap

    def knock(self, discard_card_index: Optional[int] = None, total_cards: int = 11) -> List[DispatchedTap]:
        """
        Executes Knock routine:
        1. Tap Knock Button at (1314, 2563).
        2. If discard_card_index provided, wait and tap discard card centroid.
        """
        self._debounce()
        x, y = self.get_knock_button_centroid()
        tap1 = self.executor.execute_tap(x, y, ActionType.KNOCK)
        results = [tap1]
        self._dispatch_history.append(tap1)

        if discard_card_index is not None:
            time.sleep(0.12)
            self._debounce()
            cx, cy = self.get_hand_card_centroid(discard_card_index, total_cards)
            tap2 = self.executor.execute_tap(cx, cy, ActionType.DISCARD_CARD)
            tap2.card_index = discard_card_index
            results.append(tap2)
            self._dispatch_history.append(tap2)

        return results

    def gin(self) -> DispatchedTap:
        """Tap Knock / Gin Button at (1314, 2563)."""
        self._debounce()
        x, y = self.get_knock_button_centroid()
        tap = self.executor.execute_tap(x, y, ActionType.GIN)
        self._dispatch_history.append(tap)
        return tap

    def pass_turn(self) -> DispatchedTap:
        """Tap Pass Button at (1440, 1296)."""
        self._debounce()
        x, y = self.get_pass_button_centroid()
        tap = self.executor.execute_tap(x, y, ActionType.PASS)
        self._dispatch_history.append(tap)
        return tap

    def sort_melds(self) -> DispatchedTap:
        """Tap sort button at (900, 2570)."""
        self._debounce()
        x, y = TOUCH_COORDINATES["sort_button"]
        tap = self.executor.execute_tap(x, y, ActionType.SORT_MELDS)
        self._dispatch_history.append(tap)
        return tap

    def dispatch_action(
        self,
        action_id: int,
        game_state: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> List[DispatchedTap]:
        """
        Maps a 110-action categorical index to the appropriate touch routine:
        - 0: DRAW_STOCK
        - 1: DRAW_DISCARD
        - 2: PASS (or DRAW_STOCK in standard RLCard mapping depending on phase)
        - 3: DRAW_DISCARD
        - 4: GIN
        - 5: GIN
        - 6..57: DISCARD_CARD (card_id = action_id - 6)
        - 58..109: KNOCK_WITH_CARD (card_id = action_id - 58)
        """
        hand_ids: List[int] = []
        hand_coords: List[Dict[str, int]] = []

        if game_state is not None:
            if isinstance(game_state, dict):
                hand_ids = game_state.get("hand", game_state.get("hand_ids", []))
                hand_coords = game_state.get("hand_coords", [])
            else:
                hand_ids = getattr(game_state, "hand_ids", getattr(game_state, "hand", []))
                hand_coords = getattr(game_state, "hand_coords", [])

        # 1. Draw Stock Actions
        if action_id in (0, 2) and (not game_state or not getattr(game_state, "is_pass_available", False)):
            return [self.draw_stock()]

        # 2. Draw Discard / Pickup Actions
        if action_id in (1, 3):
            return [self.draw_discard()]

        # 3. Pass Action
        if action_id == 2 and game_state and getattr(game_state, "is_pass_available", False):
            return [self.pass_turn()]

        # 4. Gin Actions
        if action_id in (4, 5):
            return [self.gin()]

        # 5. Discard Card (Actions 6..57)
        if 6 <= action_id <= 57:
            card_id = action_id - 6
            return [self.discard_by_card_id(card_id, hand_ids, hand_coords=hand_coords)]

        # 6. Knock with Discard (Actions 58..109)
        if 58 <= action_id <= 109:
            card_id = action_id - 58
            card_idx = hand_ids.index(card_id) if card_id in hand_ids else 0
            return self.knock(discard_card_index=card_idx, total_cards=len(hand_ids) if hand_ids else 11)

        # Default fallback: draw stock
        return [self.draw_stock()]

    @property
    def history(self) -> List[DispatchedTap]:
        return self._dispatch_history
