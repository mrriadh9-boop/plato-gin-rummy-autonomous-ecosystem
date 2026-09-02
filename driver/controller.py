"""
Master Bot Controller & Process Management Subsystem for Plato Gin Rummy.
Provides thread-safe start, stop, pause, resume, mode switching, and global hotkeys.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from driver.capture.scrcpy_client import ScrcpyStreamClient
from driver.dispatcher.touch_dispatcher import ADBTapDispatcher
from driver.lifecycle.match_lifecycle_manager import MatchLifecycleManager, LifecycleState, MatchStats
from driver.orchestrator import LiveDriverOrchestrator, DriverMode, LoopTelemetry
from vision.pipeline.state_aggregator import PlatoGameState

logger = logging.getLogger(__name__)


class ControllerState(str, Enum):
    """Lifecycle state of the Bot Controller."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    ERROR = "ERROR"


@dataclass
class ControllerSnapshot:
    """Atomic telemetry snapshot broadcast to UI and listeners."""
    controller_state: ControllerState = ControllerState.IDLE
    mode: DriverMode = DriverMode.AUTONOMOUS
    lifecycle_state: LifecycleState = LifecycleState.UNKNOWN
    latest_frame: Optional[np.ndarray] = None
    game_state: Optional[PlatoGameState] = None
    action_decision: Optional[str] = None
    action_type: Optional[str] = None
    action_confidence: float = 0.0
    win_probability: float = 0.50
    opponent_threat: float = 0.0
    opponent_belief: List[float] = field(default_factory=lambda: [0.19] * 52)
    telemetry: LoopTelemetry = field(default_factory=LoopTelemetry)
    stats: MatchStats = field(default_factory=MatchStats)
    status_message: str = "Ready"
    screen_off: bool = True
    fps: float = 0.0
    timestamp: float = field(default_factory=time.time)


class BotController:
    """
    Thread-safe orchestrator manager controlling the autonomous bot lifecycle.
    """

    def __init__(
        self,
        device_serial: Optional[str] = "2ace61d0",
        mode: DriverMode = DriverMode.AUTONOMOUS,
        turn_screen_off: bool = True,
        target_fps: int = 30,
        model_path: Optional[str] = None,
        use_neural: bool = True,
        auto_ranked: bool = True,
    ):
        self.device_serial = device_serial
        self.mode = mode
        self.turn_screen_off = turn_screen_off
        self.target_fps = target_fps
        self.model_path = model_path or os.path.join(PROJECT_ROOT, "ai_engine", "models", "league_latest.pt")
        self.use_neural = use_neural
        self.auto_ranked = auto_ranked

        self._state = ControllerState.IDLE
        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()

        self.subscribers: List[Callable[[ControllerSnapshot], None]] = []
        self.latest_snapshot = ControllerSnapshot(mode=self.mode, screen_off=self.turn_screen_off)
        self.orchestrator: Optional[LiveDriverOrchestrator] = None
        self.lifecycle_manager = MatchLifecycleManager()

    @property
    def state(self) -> ControllerState:
        with self._lock:
            return self._state

    def subscribe(self, callback: Callable[[ControllerSnapshot], None]) -> None:
        with self._lock:
            if callback not in self.subscribers:
                self.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[ControllerSnapshot], None]) -> None:
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)

    def _broadcast_snapshot(self, snapshot: ControllerSnapshot) -> None:
        self.latest_snapshot = snapshot
        for sub in list(self.subscribers):
            try:
                sub(snapshot)
            except Exception as e:
                logger.debug(f"[Controller] Subscriber callback error: {e}")

    def start(self) -> bool:
        with self._lock:
            if self._state == ControllerState.RUNNING:
                logger.warning("[Controller] Bot is already running.")
                return True

            self._stop_event.clear()
            self._pause_event.set()
            self._state = ControllerState.RUNNING

            self._worker_thread = threading.Thread(
                target=self._run_loop,
                name="PlatoBotWorkerThread",
                daemon=True,
            )
            self._worker_thread.start()
            logger.info("[Controller] Bot worker thread started successfully.")
            return True

    def stop(self) -> None:
        with self._lock:
            if self._state in (ControllerState.IDLE, ControllerState.STOPPING):
                return
            self._state = ControllerState.STOPPING
            self._stop_event.set()
            self._pause_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

        with self._lock:
            self._state = ControllerState.IDLE
            self._cleanup_resources()
            logger.info("[Controller] Bot stopped and cleaned up.")

    def pause(self) -> None:
        with self._lock:
            if self._state == ControllerState.RUNNING:
                self._pause_event.clear()
                self._state = ControllerState.PAUSED
                logger.info("[Controller] Bot paused.")

    def resume(self) -> None:
        with self._lock:
            if self._state == ControllerState.PAUSED:
                self._pause_event.set()
                self._state = ControllerState.RUNNING
                logger.info("[Controller] Bot resumed.")

    def toggle(self) -> None:
        if self.state == ControllerState.RUNNING:
            self.stop()
        elif self.state == ControllerState.PAUSED:
            self.resume()
        else:
            self.start()

    def set_mode(self, mode: DriverMode) -> None:
        with self._lock:
            self.mode = mode
            if self.orchestrator:
                self.orchestrator.mode = mode
            logger.info(f"[Controller] Driver mode updated to: {mode}")

    def set_screen_off(self, enabled: bool) -> None:
        with self._lock:
            self.turn_screen_off = enabled
            logger.info(f"[Controller] Screen-off setting: {enabled}")

    def _run_loop(self) -> None:
        try:
            logger.info(f"[Controller] Initializing Orchestrator (Mode: {self.mode}, Serial: {self.device_serial})...")
            
            self.orchestrator = LiveDriverOrchestrator(
                mode=self.mode,
                device_serial=self.device_serial,
                neural_model_path=self.model_path if self.use_neural else None,
                use_neural=self.use_neural,
            )

            self.orchestrator.start()
            logger.info("[Controller] Orchestrator started. Entering live frame stream loop.")

            last_fps_time = time.perf_counter()
            frame_count = 0
            current_fps = 0.0

            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                step_res = self.orchestrator.step()
                frame_count += 1

                now = time.perf_counter()
                if now - last_fps_time >= 1.0:
                    current_fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now

                frame = step_res.get("frame")
                game_state = step_res.get("state")
                action_info = step_res.get("action", {})
                telemetry = self.orchestrator.telemetry

                lifecycle_det = None
                if frame is not None and self.auto_ranked:
                    lifecycle_det = self.lifecycle_manager.detect_lifecycle_state(frame)
                    if lifecycle_det.state != LifecycleState.INGAME and self.mode == DriverMode.AUTONOMOUS:
                        self.lifecycle_manager.execute_lifecycle_action(lifecycle_det)

                act_name = action_info.get("name", "IDLE") if isinstance(action_info, dict) else str(action_info)
                act_conf = float(action_info.get("confidence", 0.0)) if isinstance(action_info, dict) else 0.0

                snapshot = ControllerSnapshot(
                    controller_state=self._state,
                    mode=self.mode,
                    lifecycle_state=lifecycle_det.state if lifecycle_det else LifecycleState.INGAME,
                    latest_frame=frame,
                    game_state=game_state,
                    action_decision=act_name,
                    action_type=act_name.split(":")[0] if ":" in act_name else act_name,
                    action_confidence=act_conf,
                    win_probability=0.50,
                    opponent_threat=float(np.mean(game_state.melds)) if game_state and game_state.melds else 0.0,
                    opponent_belief=getattr(game_state, "belief_probs", [0.19] * 52) if game_state else [0.19] * 52,
                    telemetry=telemetry,
                    stats=self.lifecycle_manager.stats,
                    status_message=f"Loop: {act_name} ({telemetry.avg_loop_time_ms:.1f}ms)",
                    screen_off=self.turn_screen_off,
                    fps=current_fps,
                    timestamp=time.time(),
                )

                self._broadcast_snapshot(snapshot)
                time.sleep(max(0.005, 1.0 / self.target_fps - (telemetry.last_loop_time_ms / 1000.0)))

        except Exception as e:
            logger.error(f"[Controller] Loop encountered error: {e}", exc_info=True)
            with self._lock:
                self._state = ControllerState.ERROR
        finally:
            self._cleanup_resources()

    def _cleanup_resources(self) -> None:
        if self.orchestrator:
            try:
                self.orchestrator.stop()
            except Exception:
                pass
            self.orchestrator = None


def setup_global_hotkeys(controller: BotController) -> None:
    try:
        import keyboard
        keyboard.add_hotkey("F9", controller.toggle)
        keyboard.add_hotkey("F10", lambda: controller.pause() if controller.state == ControllerState.RUNNING else controller.resume())
        keyboard.add_hotkey("esc", controller.stop)
        logger.info("[Hotkeys] Global hotkeys registered: F9 (Toggle Start/Stop), F10 (Pause/Resume), Esc (Stop)")
    except Exception as e:
        logger.debug(f"[Hotkeys] Could not hook global keyboard shortcuts: {e}")
