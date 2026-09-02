"""
Plato Gin Rummy Autonomous Ecosystem - Master Entry Point.
Launches the Production Desktop Application, Headless Daemon, or Copilot Overlay.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure UTF-8 stdout on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plato Gin Rummy Autonomous Engine & Production Cockpit"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["AUTONOMOUS", "COPILOT", "EVAL", "MANUAL"],
        default="AUTONOMOUS",
        help="Bot execution mode (default: AUTONOMOUS)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless without GUI (daemon mode)",
    )
    parser.add_argument(
        "--device-serial",
        type=str,
        default="2ace61d0",
        help="Target ADB device serial (default: 2ace61d0)",
    )
    parser.add_argument(
        "--turn-screen-off",
        action="store_true",
        default=True,
        help="Turn off tablet display while playing (default: True)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target streaming loop FPS (default: 30)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.debug)

    print("========================================================")
    print("  PLATO GIN RUMMY AUTONOMOUS ECOSYSTEM (v2.0 PRO)")
    print(f"  Target Device: {args.device_serial} (1800x2880 Canvas)")
    print(f"  Mode:          {args.mode}")
    print(f"  Screen Off:    {args.turn_screen_off}")
    print("========================================================")

    if args.headless:
        # Headless Autonomous Daemon
        from driver.controller import BotController, setup_global_hotkeys
        from driver.orchestrator import DriverMode

        mode_enum = getattr(DriverMode, args.mode, DriverMode.AUTONOMOUS)
        controller = BotController(
            device_serial=args.device_serial,
            mode=mode_enum,
            turn_screen_off=args.turn_screen_off,
            target_fps=args.fps,
            auto_ranked=True,
        )
        setup_global_hotkeys(controller)

        print("[Daemon] Starting autonomous ranked loop. Press Ctrl+C or Esc to stop.")
        controller.start()

        try:
            import time
            while controller.state in (controller.state.RUNNING, controller.state.PAUSED):
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[Daemon] Stop requested by user.")
        finally:
            controller.stop()
            print("[Daemon] Clean shutdown complete.")
    else:
        # Launch Production Desktop Cockpit (PyQt6)
        from PyQt6.QtWidgets import QApplication
        from driver.hud.production_app import PlatoBotProApp
        from driver.controller import setup_global_hotkeys

        app = QApplication(sys.argv)
        app.setApplicationName("PlatoBotPro")
        window = PlatoBotProApp(
            device_serial=args.device_serial,
            turn_screen_off=args.turn_screen_off,
        )
        setup_global_hotkeys(window.controller)
        window.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    main()
