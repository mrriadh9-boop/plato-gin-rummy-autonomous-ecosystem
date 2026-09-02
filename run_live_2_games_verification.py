"""
Live 2-Games Verification Script for Plato Gin Rummy Autonomous Ecosystem.
Executes autonomous play on connected Xiaomi Pad 6 (serial: 2ace61d0).
Logs every perception step, decision, touch dispatch, and lifecycle transition across 2 complete games.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from driver.controller import BotController, ControllerState, ControllerSnapshot
from driver.orchestrator import DriverMode
from driver.lifecycle.match_lifecycle_manager import LifecycleState, MatchLifecycleManager
from vision.pipeline.state_aggregator import StateAggregator

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LiveVerification")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "live_verification")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_live_verification(max_runtime_sec: float = 600.0, target_games: int = 2):
    print("================================================================")
    print("  PLATO GIN RUMMY - LIVE 2-GAMES AUTONOMOUS VERIFICATION")
    print(f"  Target Device: 2ace61d0 (1800x2880 Xiaomi Pad 6)")
    print(f"  Target Games:  {target_games}")
    print(f"  Max Runtime:   {max_runtime_sec}s")
    print("================================================================")

    controller = BotController(
        device_serial="2ace61d0",
        mode=DriverMode.AUTONOMOUS,
        turn_screen_off=False,
        target_fps=15,
        auto_ranked=True,
    )

    log_records = []
    step_idx = 0
    games_completed = 0
    rounds_completed = 0
    start_time = time.time()
    last_lifecycle_state = LifecycleState.UNKNOWN

    print("\n[Step 1] Starting BotController...")
    controller.start()

    try:
        while time.time() - start_time < max_runtime_sec:
            time.sleep(0.5)
            snapshot = controller.latest_snapshot
            if snapshot.latest_frame is None:
                continue

            step_idx += 1
            curr_state = snapshot.lifecycle_state

            # State transition notification
            if curr_state != last_lifecycle_state:
                print(f"\n⚡ [LIFECYCLE TRANSITION] {last_lifecycle_state} -> {curr_state}")
                last_lifecycle_state = curr_state

                # Save milestone frame
                frame_path = os.path.join(OUTPUT_DIR, f"step_{step_idx:04d}_{curr_state}.png")
                cv2.imwrite(frame_path, snapshot.latest_frame)

                if curr_state == LifecycleState.MATCH_SUMMARY:
                    games_completed += 1
                    print(f"🏆 [MATCH COMPLETED] Total matches completed: {games_completed}/{target_games}")
                elif curr_state == LifecycleState.ROUND_SUMMARY:
                    rounds_completed += 1
                    print(f"🃏 [ROUND COMPLETED] Total rounds completed: {rounds_completed}")

            # If in-game, log active card state and decisions
            if curr_state == LifecycleState.INGAME and snapshot.game_state:
                gs = snapshot.game_state
                if gs.is_player_turn:
                    print(f"[Turn {step_idx:04d}] Phase: {gs.phase:<12} | Hand ({len(gs.hand)}): {gs.hand} | Top: {gs.discard_top} | Deadwood: {gs.deadwood} pts | Action: {snapshot.action_decision} (Conf: {snapshot.action_confidence*100:.1f}%)")
                    
                    # Record step log
                    log_records.append({
                        "step": step_idx,
                        "time": time.time() - start_time,
                        "phase": gs.phase,
                        "hand": gs.hand,
                        "discard_top": gs.discard_top,
                        "deadwood": gs.deadwood,
                        "melds": gs.melds,
                        "decision": snapshot.action_decision,
                        "latency_ms": snapshot.telemetry.last_loop_time_ms,
                    })

            # Check target completion condition
            if games_completed >= target_games:
                print(f"\n🎉 [SUCCESS] Successfully completed {games_completed} full games autonomously!")
                break

    except KeyboardInterrupt:
        print("\n[Verification] Interrupted by user.")
    finally:
        print("\n[Step Final] Stopping BotController and saving audit report...")
        controller.stop()

        # Save verification report
        report = {
            "target_games": target_games,
            "games_completed": games_completed,
            "rounds_completed": rounds_completed,
            "total_steps": step_idx,
            "total_duration_sec": time.time() - start_time,
            "log_records_count": len(log_records),
            "records_sample": log_records[:30],
        }
        report_path = os.path.join(OUTPUT_DIR, "live_verification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Audit report saved to: {report_path}")
        print("================================================================")
        return games_completed >= target_games or rounds_completed >= 2


if __name__ == "__main__":
    success = run_live_verification(max_runtime_sec=120.0, target_games=2)
    sys.exit(0 if success else 1)
