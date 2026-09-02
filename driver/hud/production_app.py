"""
Plato Bot Pro - Production Desktop Application for Autonomous Ranked Matches.
Provides high-FPS hardware video mirroring, real-time card perception HUD,
opponent belief heatmaps, decision analytics, and complete session controls.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QGradient,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from driver.controller import BotController, ControllerSnapshot, ControllerState
from driver.orchestrator import DriverMode
from driver.lifecycle.match_lifecycle_manager import LifecycleState
from vision.pipeline.state_aggregator import PlatoGameState

logger = logging.getLogger(__name__)

# Canonical Ranks and Suits
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["♠", "♥", "♦", "♣"]


class LiveVideoCanvas(QWidget):
    """
    Renders live 1800x2880 tablet video stream with card overlays, melds, and touch indicators.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(360, 576)
        self.latest_frame: Optional[np.ndarray] = None
        self.game_state: Optional[PlatoGameState] = None
        self.last_tap_point: Optional[Tuple[int, int]] = None
        self.tap_animation_radius: float = 0.0
        self._pixmap: Optional[QPixmap] = None

    def update_frame(self, frame: Optional[np.ndarray], state: Optional[PlatoGameState], tap_point: Optional[Tuple[int, int]] = None):
        self.latest_frame = frame
        self.game_state = state
        if tap_point:
            self.last_tap_point = tap_point
            self.tap_animation_radius = 10.0
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(14, 16, 22))

        if self.latest_frame is None:
            painter.setPen(QColor(100, 116, 139))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "Waiting for Tablet Stream...\n(Device: 2ace61d0 / Xiaomi Pad 6)")
            return

        # Convert BGR frame to QImage / QPixmap
        fh, fw = self.latest_frame.shape[:2]
        bytes_per_line = 3 * fw
        q_img = QImage(self.latest_frame.data, fw, fh, bytes_per_line, QImage.Format.Format_BGR888)
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        pw, ph = scaled_pixmap.width(), scaled_pixmap.height()
        ox = (w - pw) // 2
        oy = (h - ph) // 2
        painter.drawPixmap(ox, oy, scaled_pixmap)

        # Draw Overlay Annotations (scale from 1800x2880 to pw x ph)
        scale_x = pw / 1800.0
        scale_y = ph / 2880.0

        if self.game_state and self.game_state.hand:
            # Draw Hand Card Bounding Boxes and Labels
            melded_set = set()
            if self.game_state.melds:
                for m in self.game_state.melds:
                    for c in m:
                        melded_set.add(c)

            hand_count = len(self.game_state.hand)
            step = 114.7
            total_fan_w = (hand_count - 1) * step + 260.0
            start_x = (1800 - total_fan_w) / 2.0

            for i, card_id in enumerate(self.game_state.hand):
                card_x = start_x + i * step
                rx = int(ox + card_x * scale_x)
                ry = int(oy + 1780 * scale_y)
                rw = int((step if i < hand_count - 1 else 260.0) * scale_x)
                rh = int(480 * scale_y)

                is_melded = card_id in melded_set
                border_color = QColor(0, 230, 118, 220) if is_melded else QColor(255, 171, 0, 180)
                bg_color = QColor(0, 230, 118, 40) if is_melded else QColor(255, 171, 0, 25)

                painter.setPen(QPen(border_color, 2))
                painter.setBrush(QBrush(bg_color))
                painter.drawRoundedRect(rx, ry, rw, rh, 6, 6)

                # Card Label Pill
                pill_rect = QRect(rx + 2, ry - int(24 * scale_y), max(36, rw - 4), int(22 * scale_y))
                painter.setBrush(QBrush(QColor(20, 24, 33, 230)))
                painter.setPen(QPen(border_color, 1))
                painter.drawRoundedRect(pill_rect, 4, 4)

                painter.setFont(QFont("Segoe UI", max(8, int(11 * scale_x)), QFont.Weight.Bold))
                text_color = QColor(255, 82, 82) if any(s in card_id for s in ["♥", "♦"]) else QColor(240, 240, 240)
                painter.setPen(text_color)
                painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, card_id)

        # Draw Top Discard Pill
        if self.game_state and self.game_state.discard_top:
            dx = int(ox + 1080 * scale_x)
            dy = int(oy + 1290 * scale_y)
            painter.setPen(QPen(QColor(0, 210, 255), 2))
            painter.setBrush(QBrush(QColor(0, 210, 255, 30)))
            painter.drawRoundedRect(dx - 40, dy - 50, 80, 100, 8, 8)
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(QRect(dx - 40, dy - 50, 80, 100), Qt.AlignmentFlag.AlignCenter, f"TOP:\n{self.game_state.discard_top}")


class OpponentBeliefWidget(QWidget):
    """
    Renders 4x13 heat matrix of neural probabilities predicting opponent hand cards.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.belief_probs: List[float] = [0.19] * 52
        self.setFixedHeight(120)

    def update_belief(self, probs: List[float]):
        self.belief_probs = probs if len(probs) == 52 else [0.19] * 52
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(20, 24, 33))

        cell_w = (w - 30) / 13.0
        cell_h = (h - 20) / 4.0

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))

        for s_idx, suit in enumerate(SUITS):
            # Suit label
            sy = int(10 + s_idx * cell_h)
            painter.setPen(QColor(255, 82, 82) if suit in ["♥", "♦"] else QColor(200, 200, 200))
            painter.drawText(QRect(2, sy, 22, int(cell_h)), Qt.AlignmentFlag.AlignCenter, suit)

            for r_idx, rank in enumerate(RANKS):
                card_idx = s_idx * 13 + r_idx
                prob = self.belief_probs[card_idx] if card_idx < len(self.belief_probs) else 0.0

                cx = int(26 + r_idx * cell_w)
                cy = sy

                # Color intensity based on probability: 0 (dark) -> 1.0 (bright amber/cyan)
                alpha = int(np.clip(prob * 255 * 3.0, 15, 240))
                color = QColor(0, 210, 255, alpha) if prob > 0.40 else QColor(40, 50, 70, alpha)

                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(30, 36, 48), 1))
                painter.drawRoundedRect(cx, cy, int(cell_w - 2), int(cell_h - 2), 2, 2)

                if cell_w > 18:
                    painter.setPen(QColor(255, 255, 255, 200) if prob > 0.30 else QColor(140, 150, 170, 180))
                    painter.drawText(QRect(cx, cy, int(cell_w - 2), int(cell_h - 2)), Qt.AlignmentFlag.AlignCenter, rank)


class PlatoBotProApp(QMainWindow):
    """
    Complete production desktop application window for Plato Gin Rummy Autonomous System.
    """

    snapshot_signal = pyqtSignal(object)

    def __init__(self, device_serial: str = "2ace61d0", turn_screen_off: bool = True):
        super().__init__()
        self.setWindowTitle("⚡ PLATO BOT PRO - AUTONOMOUS RANKED ENGINE")
        self.resize(1280, 850)
        self.setMinimumSize(1000, 700)

        # Bot Controller Instance
        self.controller = BotController(
            device_serial=device_serial,
            mode=DriverMode.AUTONOMOUS,
            turn_screen_off=turn_screen_off,
            auto_ranked=True,
        )

        self.snapshot_signal.connect(self._on_snapshot_received)
        self.controller.subscribe(lambda s: self.snapshot_signal.emit(s))

        self._setup_ui()
        self._apply_theme()

        # Update Timer for Clock and UI Polish
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_ui_tick)
        self.timer.start(500)

    def _setup_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # 1. Top Header Bar
        header = QFrame()
        header.setObjectName("headerFrame")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(16, 10, 16, 10)

        title_box = QVBoxLayout()
        title_lbl = QLabel("⚡ PLATO GIN RUMMY AUTONOMOUS BOT")
        title_lbl.setObjectName("titleLabel")
        sub_lbl = QLabel(f"Connected: Xiaomi Pad 6 [{self.controller.device_serial}] (1800x2880 Canvas)")
        sub_lbl.setObjectName("subLabel")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        h_layout.addLayout(title_box)

        h_layout.addStretch()

        # Status Badge
        self.status_badge = QLabel("● IDLE")
        self.status_badge.setObjectName("statusBadgeIdle")
        h_layout.addWidget(self.status_badge)

        # Action Buttons
        self.btn_start = QPushButton("▶ START (F9)")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.controller.start)
        h_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸ PAUSE (F10)")
        self.btn_pause.setObjectName("btnPause")
        self.btn_pause.clicked.connect(self._on_pause_clicked)
        h_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹ STOP (Esc)")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.clicked.connect(self.controller.stop)
        h_layout.addWidget(self.btn_stop)

        root_layout.addWidget(header)

        # 2. Main Content Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # Left Panel: Live Stream Canvas & Quick Toggles
        left_container = QWidget()
        l_layout = QVBoxLayout(left_container)
        l_layout.setContentsMargins(0, 0, 0, 0)
        l_layout.setSpacing(8)

        self.canvas = LiveVideoCanvas()
        l_layout.addWidget(self.canvas, stretch=1)

        # Canvas Controls Bar
        canvas_bar = QHBoxLayout()
        self.cb_screen_off = QCheckBox("Keep Tablet Screen Off (--turn-screen-off)")
        self.cb_screen_off.setChecked(self.controller.turn_screen_off)
        self.cb_screen_off.toggled.connect(self.controller.set_screen_off)
        canvas_bar.addWidget(self.cb_screen_off)

        self.cb_auto_ranked = QCheckBox("Auto-Queue Ranked Matches")
        self.cb_auto_ranked.setChecked(True)
        self.cb_auto_ranked.toggled.connect(lambda v: setattr(self.controller, "auto_ranked", v))
        canvas_bar.addWidget(self.cb_auto_ranked)

        canvas_bar.addStretch()
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setObjectName("metricLabel")
        canvas_bar.addWidget(self.fps_label)

        l_layout.addLayout(canvas_bar)
        splitter.addWidget(left_container)

        # Right Panel: Decision Cockpit, Belief Heatmap & Telemetry
        right_container = QWidget()
        r_layout = QVBoxLayout(right_container)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(10)

        # AI Decision Card
        dec_box = QGroupBox("AI DECISION & REASONING COCKPIT")
        dec_layout = QVBoxLayout(dec_box)

        self.action_label = QLabel("ACTION: WAITING FOR TURN")
        self.action_label.setObjectName("actionLabel")
        dec_layout.addWidget(self.action_label)

        self.provenance_label = QLabel("Strategy: Neural GRU Policy + IS-MCTS Search")
        self.provenance_label.setObjectName("provenanceLabel")
        dec_layout.addWidget(self.provenance_label)

        # Win Probability Bar
        win_layout = QHBoxLayout()
        win_layout.addWidget(QLabel("Win Rate Est:"))
        self.win_bar = QProgressBar()
        self.win_bar.setRange(0, 100)
        self.win_bar.setValue(50)
        self.win_bar.setTextVisible(True)
        self.win_bar.setFormat("%v%")
        win_layout.addWidget(self.win_bar)
        dec_layout.addLayout(win_layout)

        # Deadwood comparison
        self.dw_label = QLabel("Deadwood: 0 (Target: Knock <= 10, Gin = 0)")
        self.dw_label.setObjectName("metricLabel")
        dec_layout.addWidget(self.dw_label)

        r_layout.addWidget(dec_box)

        # Opponent Belief Matrix
        belief_box = QGroupBox("OPPONENT PRIVATE HAND BELIEF HEATMAP (52 Cards)")
        b_layout = QVBoxLayout(belief_box)
        self.belief_widget = OpponentBeliefWidget()
        b_layout.addWidget(self.belief_widget)
        r_layout.addWidget(belief_box)

        # Session & Ranked Telemetry
        stats_box = QGroupBox("SESSION & RANKED MATCH TELEMETRY")
        s_layout = QGridLayout(stats_box)

        self.stat_matches = QLabel("Matches Played: 0")
        self.stat_winrate = QLabel("Win Rate: 0.0%")
        self.stat_rounds = QLabel("Hands Played: 0")
        self.stat_latency = QLabel("Avg Decision: 0.0 ms")

        s_layout.addWidget(self.stat_matches, 0, 0)
        s_layout.addWidget(self.stat_winrate, 0, 1)
        s_layout.addWidget(self.stat_rounds, 1, 0)
        s_layout.addWidget(self.stat_latency, 1, 1)

        r_layout.addWidget(stats_box)

        # Mode Selector & Settings
        mode_box = QGroupBox("EXECUTION MODE & SETTINGS")
        m_layout = QHBoxLayout(mode_box)

        m_layout.addWidget(QLabel("Driver Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["AUTONOMOUS (Full Auto Ranked)", "COPILOT (Visual Advice Only)", "MANUAL (Passive State Tracking)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        m_layout.addWidget(self.combo_mode)

        r_layout.addWidget(mode_box)

        # Real-Time Event Log
        log_box = QGroupBox("EVENT & AUDIT LOG")
        log_layout = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(130)
        log_layout.addWidget(self.log_text)
        r_layout.addWidget(log_box)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root_layout.addWidget(splitter, stretch=1)

    def _apply_theme(self):
        style = """
        QMainWindow, QWidget {
            background-color: #0d1117;
            color: #e6edf3;
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        #headerFrame {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #161b22, stop:1 #1f242c);
            border: 1px solid #30363d;
            border-radius: 10px;
        }
        #titleLabel {
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
            letter-spacing: 0.5px;
        }
        #subLabel {
            font-size: 12px;
            color: #8b949e;
        }
        QGroupBox {
            font-size: 12px;
            font-weight: bold;
            color: #79c0ff;
            border: 1px solid #30363d;
            border-radius: 8px;
            margin-top: 10px;
            padding-top: 14px;
            background-color: #161b22;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        #statusBadgeIdle {
            background-color: #21262d;
            color: #8b949e;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            border: 1px solid #30363d;
        }
        #statusBadgeRunning {
            background-color: #0e4429;
            color: #3fb950;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            border: 1px solid #238636;
        }
        #statusBadgePaused {
            background-color: #4d2d00;
            color: #d29922;
            padding: 6px 14px;
            border-radius: 6px;
            font-weight: bold;
            font-size: 12px;
            border: 1px solid #9e6a03;
        }
        #btnStart {
            background-color: #238636;
            color: white;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 18px;
            border-radius: 6px;
            border: none;
        }
        #btnStart:hover { background-color: #2ea043; }
        #btnPause {
            background-color: #9e6a03;
            color: white;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 14px;
            border-radius: 6px;
            border: none;
        }
        #btnPause:hover { background-color: #bb8009; }
        #btnStop {
            background-color: #da3633;
            color: white;
            font-weight: bold;
            font-size: 12px;
            padding: 8px 14px;
            border-radius: 6px;
            border: none;
        }
        #btnStop:hover { background-color: #f85149; }
        #actionLabel {
            font-size: 15px;
            font-weight: bold;
            color: #58a6ff;
            background-color: #0d1117;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
        }
        #provenanceLabel {
            font-size: 11px;
            color: #8b949e;
        }
        #metricLabel {
            font-size: 12px;
            color: #c9d1d9;
        }
        QProgressBar {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 4px;
            text-align: center;
            color: white;
            height: 16px;
        }
        QProgressBar::chunk {
            background-color: #1f6feb;
            border-radius: 3px;
        }
        QComboBox, QCheckBox {
            background-color: #21262d;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 4px 8px;
            color: #c9d1d9;
        }
        QTextEdit {
            background-color: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            font-family: 'Consolas', monospace;
            font-size: 11px;
            color: #8b949e;
        }
        """
        self.setStyleSheet(style)

    @pyqtSlot(object)
    def _on_snapshot_received(self, snapshot: ControllerSnapshot):
        # 1. Update Video Canvas
        self.canvas.update_frame(snapshot.latest_frame, snapshot.game_state)

        # 2. Update Status Badge
        if snapshot.controller_state == ControllerState.RUNNING:
            self.status_badge.setText("● AUTONOMOUS RUNNING")
            self.status_badge.setObjectName("statusBadgeRunning")
        elif snapshot.controller_state == ControllerState.PAUSED:
            self.status_badge.setText("● PAUSED")
            self.status_badge.setObjectName("statusBadgePaused")
        else:
            self.status_badge.setText("● IDLE")
            self.status_badge.setObjectName("statusBadgeIdle")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

        # 3. Update AI Decision
        if snapshot.action_decision:
            self.action_label.setText(f"ACTION: {snapshot.action_decision} (Conf: {snapshot.action_confidence*100:.1f}%)")

        # 4. Update Win Probability & Deadwood
        if snapshot.game_state:
            dw = snapshot.game_state.deadwood
            self.dw_label.setText(f"Deadwood: {dw} pts | Target: Knock <= 10, Gin = 0")
            win_pct = int(np.clip((100 - dw * 3.0), 5, 95))
            self.win_bar.setValue(win_pct)

        # 5. Update Belief Heatmap
        if snapshot.opponent_belief:
            self.belief_widget.update_belief(snapshot.opponent_belief)

        # 6. Update Stats
        stats = snapshot.stats
        self.stat_matches.setText(f"Matches Played: {stats.matches_played} (Won: {stats.matches_won})")
        self.stat_winrate.setText(f"Win Rate: {stats.win_rate:.1f}%")
        self.stat_rounds.setText(f"Hands Played: {stats.total_rounds_played}")
        self.stat_latency.setText(f"Avg Decision: {snapshot.telemetry.avg_loop_time_ms:.1f} ms")
        self.fps_label.setText(f"FPS: {snapshot.fps:.1f}")

        # 7. Append log if state transition
        if snapshot.lifecycle_state != LifecycleState.INGAME:
            self.log_text.append(f"[{time.strftime('%H:%M:%S')}] Lifecycle: {snapshot.lifecycle_state} -> {snapshot.status_message}")

    def _on_pause_clicked(self):
        if self.controller.state == ControllerState.RUNNING:
            self.controller.pause()
        elif self.controller.state == ControllerState.PAUSED:
            self.controller.resume()

    def _on_mode_changed(self, idx: int):
        modes = [DriverMode.AUTONOMOUS, DriverMode.COPILOT, DriverMode.MANUAL]
        if idx < len(modes):
            self.controller.set_mode(modes[idx])

    def _on_ui_tick(self):
        pass

    def closeEvent(self, event):
        self.controller.stop()
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plato Bot Pro Desktop Application")
    parser.add_argument("--device-serial", type=str, default="2ace61d0", help="ADB Device Serial")
    parser.add_argument("--turn-screen-off", action="store_true", default=True, help="Keep tablet screen off during play")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = PlatoBotProApp(device_serial=args.device_serial, turn_screen_off=args.turn_screen_off)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
