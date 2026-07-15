#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════╗
║  Main Entry Point — Autonomous Vehicle Simulation             ║
║  Diffusion Policy Pipeline dengan PyBullet                     ║
╚════════════════════════════════════════════════════════════════╝

Cara menjalankan:
    python main.py

Keyboard:
  C  - Toggle action chunking
  Q  - Quit
"""

import pybullet as p
import time

from av_config import *
from av_input_simple import InputHandler
from av_vehicle import Racecar
from av_environment import Environment
from av_controller import PurePursuitController
from av_simulator import DiffusionPolicySimulator


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  Autonomous Vehicle — Diffusion Policy Pipeline")
    print("  PyBullet Physics Simulation")
    print("=" * 70)
    print()
    print("  Pipeline: Observation → Encoder → Diffusion → Action → Control")
    print()
    print("  Controls:")
    print("    C  Toggle Action Chunking")
    print("    Q  Quit")
    print()
    print("=" * 70)
    print()


def main():
    """Main entry point."""
    print_banner()

    # Initialize
    print("[INIT] Initializing environment...")
    env = Environment(use_gui=ENABLE_GUI)

    print("[INIT] Loading vehicle model...")
    vehicle = Racecar(start_pos=[-3.5, -3.5, 0.1])

    print("[INIT] Setting up controller...")
    controller = PurePursuitController(env.waypoints, lookahead=1.5, speed=40.0)

    print("[INIT] Initializing input handler...")
    input_handler = InputHandler()

    print("[INIT] Creating simulator...")
    sim = DiffusionPolicySimulator(env, vehicle, controller, input_handler)

    # Camera viewpoint
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
    )

    print("[INFO] Simulation ready. Autonomous navigation starting...")
    print()

    # Main loop
    try:
        while sim.running:
            sim.step()
            time.sleep(SIMULATION_STEP)

            # Check quit via toggles
            _, _, toggles = input_handler.get_action()
            if toggles.get("quit"):
                print("\n[INFO] Quit. Shutting down...")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[CLEANUP] Cleaning up...")
        sim.cleanup()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
