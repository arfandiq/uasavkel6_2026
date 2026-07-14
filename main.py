#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════╗
║  Main Entry Point — Autonomous Vehicle Simulation             ║
║  Diffusion Policy Demo dengan PyBullet                         ║
║  Support: Keyboard + Xbox Controller Input                     ║
║  GPU Acceleration (Nvidia CUDA)                                ║
╚════════════════════════════════════════════════════════════════╝

Cara menjalankan:
    python main.py

Atau install dependencies dulu:
    pip install pybullet numpy pygame

Keyboard Control:
  WASD/Arrows    - Drive
  SPACE          - Toggle mode (MANUAL ↔ AUTONOMOUS)
  R              - Record demonstrasi
  P              - Replay demonstrasi
  C              - Toggle action chunking
  1/2            - Control mode (position/velocity)
  Q              - Quit

Xbox Controller:
  Left Stick X   - Steering
  Triggers       - Throttle (RT forward, LT backward)
  D-Pad          - Alternative input
  Buttons        - Toggle modes (when implemented)
"""

import pybullet as p
import time
import sys

from av_config import *
from av_input_simple import InputHandler
from av_vehicle import Racecar
from av_environment import Environment
from av_controller import PurePursuitController
from av_simulator import DiffusionPolicySimulator


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  Autonomous Vehicle Simulation — Diffusion Policy Demo        ║")
    print("║  PyBullet Physics Engine with GPU Acceleration (Nvidia CUDA)  ║")
    print("║  Support: Keyboard + Xbox Controller                           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print("=" * 70)
    print()
    print("PAPER CONCEPTS DITAMPILKAN:")
    print("  1. Behavior Cloning      → Record & Replay demonstrasi")
    print("  2. Action Chunking       → Prediksi N langkah sekaligus")
    print("  3. Position vs Velocity  → Perbandingan mode kontrol")
    print("  4. Autonomous Navigation → Waypoint-based pure pursuit")
    print()
    print("KEYBOARD CONTROLS:")
    print("  WASD / Arrow Keys  - Gerakkan mobil")
    print("  SPACE              - Toggle mode (MANUAL ↔ AUTONOMOUS)")
    print("  R                  - Record demonstrasi (Behavior Cloning)")
    print("  P                  - Replay demonstrasi")
    print("  C                  - Toggle Action Chunking")
    print("  1                  - Position Control mode")
    print("  2                  - Velocity Control mode")
    print("  Q                  - Quit")
    print()
    print("XBOX CONTROLLER:")
    print("  Left Stick X       - Steering")
    print("  Triggers (L/R)     - Throttle")
    print()
    print("=" * 70)
    print()


def main():
    """Main entry point untuk simulasi."""
    print_banner()

    # Initialize components
    print("[INIT] Initializing environment...")
    env = Environment(use_gui=ENABLE_GUI)

    print("[INIT] Loading vehicle model...")
    vehicle = Racecar(start_pos=[-3.5, -3.5, 0.1])

    print("[INIT] Setting up controller...")
    controller = PurePursuitController(env.waypoints, lookahead=1.5, speed=40.0)

    print("[INIT] Initializing input handler...")
    input_handler = InputHandler(input_mode=INPUT_BOTH)

    print("[INIT] Creating simulator...")
    sim = DiffusionPolicySimulator(env, vehicle, controller, input_handler)

    # Camera viewpoint (fixed, no mouse control)
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0],
    )

    print("[INFO] ✅ Simulasi siap! Tekan kontrol untuk mulai...")
    print()

    # Main loop
    try:
        while sim.running:
            sim.step()
            time.sleep(SIMULATION_STEP)

            # Check for quit command
            try:
                keys = p.getKeyboardEvents()
                if ord('q') in keys or ord('Q') in keys:
                    print("\n[INFO] Quit command received. Shutting down...")
                    break
            except:
                pass

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] Simulasi error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[CLEANUP] Cleaning up resources...")
        sim.cleanup()
        print("[INFO] ✅ Simulasi selesai. Terima kasih!")
        print()


if __name__ == "__main__":
    main()
