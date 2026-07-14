"""
╔════════════════════════════════════════════════════════════════╗
║  Diffusion Policy Simulator — Main Simulator Class            ║
║  Menggabungkan semua komponen untuk simulasi autonomous vehicle║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p
import numpy as np
import time
from collections import deque
from av_config import *
from av_input_simple import InputHandler
from av_vehicle import Racecar
from av_environment import Environment
from av_controller import PurePursuitController


class DiffusionPolicySimulator:
    """Simulator yang mendemonstrasikan konsep-konsep dari paper Diffusion Policy.

    Konsep yang disimulasikan:
    1. BEHAVIOR CLONING → Record demonstrasi manual → replay otomatis
    2. ACTION CHUNKING   → Prediksi & eksekusi N aksi sekaligus
    3. DENOISING         → Smoothing dan refinement trajectory
    4. POSITION CONTROL  → Lebih stabil (temuan paper)
    """

    def __init__(self, env, vehicle, controller, input_handler):
        self.env = env
        self.vehicle = vehicle
        self.controller = controller
        self.input_handler = input_handler

        # State
        self.mode = MODE_AUTONOMOUS
        self.control_mode = CTRL_POSITION
        self.running = True
        self.use_action_chunking = True

        # Demonstration data (Behavior Cloning)
        self.is_recording = False
        self.demonstrations = []
        self.current_demo = []

        # Replay
        self.is_replaying = False
        self.replay_idx = 0
        self.replay_data = None

        # Action chunking buffer
        self.action_chunk_buffer = deque(maxlen=ACTION_CHUNK_SIZE)

        # Trajectory visualization
        self.trajectory_spheres = []

        # Stats
        self.step_count = 0
        self.goal_reached = False

    def step(self):
        """Satu langkah simulasi."""
        if self.step_count >= MAX_SIMULATION_STEPS:
            self.running = False
            return

        # --- Input Handling ---
        throttle, steering, toggles = self.input_handler.get_action()

        # Proses toggles
        self._handle_toggles(toggles)

        # --- Step Logic ---
        if self.mode == MODE_MANUAL or self.is_replaying:
            if self.is_replaying:
                self._step_replay()
            else:
                self._step_manual(throttle, steering)
        else:
            self._step_autonomous()

        # --- Step Physics ---
        p.stepSimulation()

        # --- Cek goal ---
        if not self.goal_reached:
            state = self.vehicle.get_state()
            goal_pos = self.controller.waypoints[-1][:2]
            dist_to_goal = np.linalg.norm(state["position"][:2] - goal_pos)
            if dist_to_goal < 0.5:
                self.goal_reached = True
                print(f"[INFO] 🎯 GOAL TERCAPAI dalam {self.step_count} steps!")

        self.step_count += 1

        # --- UI Text ---
        self._draw_ui()

    def _handle_toggles(self, toggles):
        """Handle toggle commands dari input handler."""
        if toggles.get("space"):
            self.mode = MODE_MANUAL if self.mode == MODE_AUTONOMOUS else MODE_AUTONOMOUS
            print(f"[INFO] Mode: {self.mode}")
            if self.mode == MODE_AUTONOMOUS:
                self.controller.reset()

        if toggles.get("record"):
            self.is_recording = not self.is_recording
            if self.is_recording:
                self.current_demo = []
                print("[INFO] ▶ MULAI MEREKAM demonstrasi (Behavior Cloning)...")
            else:
                if len(self.current_demo) > 10:
                    self.demonstrations.append(self.current_demo)
                    print(f"[INFO] ⏹ Demonstrasi selesai! {len(self.current_demo)} steps tersimpan.")
                    print(f"       Total demo tersimpan: {len(self.demonstrations)}")
                else:
                    print("[INFO] Demonstrasi terlalu pendek, dibatalkan.")

        if toggles.get("replay"):
            if len(self.demonstrations) > 0 and not self.is_replaying:
                self.is_replaying = True
                self.replay_idx = 0
                self.replay_data = self.demonstrations[-1]
                self.vehicle.reset()
                self.controller.reset()
                print(f"[INFO] ▶ REPLAY demonstrasi ({len(self.replay_data)} steps)")
            else:
                self.is_replaying = False
                self.replay_data = None
                print("[INFO] ⏹ Replay dihentikan.")

        if toggles.get("chunking"):
            self.use_action_chunking = not self.use_action_chunking
            print(f"[INFO] Action Chunking: {'AKTIF' if self.use_action_chunking else 'NONAKTIF'}")

        if toggles.get("pos_ctrl"):
            self.control_mode = CTRL_POSITION
            print(f"[INFO] Control mode: POSITION CONTROL")

        if toggles.get("vel_ctrl"):
            self.control_mode = CTRL_VELOCITY
            print(f"[INFO] Control mode: VELOCITY CONTROL")

    def _step_manual(self, throttle, steering):
        """Step untuk mode manual."""
        self.vehicle.apply_action(throttle, steering, self.control_mode)

        if self.is_recording:
            state = self.vehicle.get_state()
            obs = {
                "position": state["position"].copy(),
                "yaw": state["yaw"],
                "speed": state["speed"],
            }
            action = np.array([throttle, steering])
            self.current_demo.append((obs, action))

    def _step_autonomous(self):
        """Step untuk mode autonomous (waypoint following)."""
        state = self.vehicle.get_state()
        throttle, steering = self.controller.compute_action(state)

        # ACTION CHUNKING
        if self.use_action_chunking:
            self.action_chunk_buffer.append([throttle, steering])
            if len(self.action_chunk_buffer) >= ACTION_CHUNK_SIZE:
                chunk = np.mean(self.action_chunk_buffer, axis=0)
                throttle, steering = chunk[0], chunk[1]

        self.vehicle.apply_action(throttle, steering, self.control_mode)
        self._visualize_trajectory_preview(state)

        # ===== WAYPOINT STATUS OUTPUT =====
        current_wp_idx = self.controller.current_wp_idx
        total_waypoints = len(self.controller.waypoints)

        # Print status setiap frame
        if current_wp_idx < total_waypoints:
            current_wp = self.controller.waypoints[current_wp_idx]
            dist_to_wp = np.linalg.norm(state["position"][:2] - current_wp[:2])
            progress = f"{current_wp_idx}/{total_waypoints-1}"
            print(f"[AUTONOMOUS] Heading to WP{progress} | Distance: {dist_to_wp:.2f}m | Speed: {state['speed']:.2f} m/s")
        else:
            print(f"[AUTONOMOUS] ALL WAYPOINTS COMPLETED!")

    def _step_replay(self):
        """Replay demonstrasi (Behavior Cloning)."""
        if self.replay_data is None or self.replay_idx >= len(self.replay_data):
            print("[INFO] ✅ Replay selesai!")
            self.is_replaying = False
            return

        _, action = self.replay_data[self.replay_idx]
        self.vehicle.apply_action(action[0], action[1], self.control_mode)
        self.replay_idx += 1

    def _visualize_trajectory_preview(self, state):
        """Visualisasi predicted trajectory."""
        self.env.cleanup_debug(self.trajectory_spheres)
        self.trajectory_spheres = []

        if self.use_action_chunking:
            upcoming = self.controller.get_upcoming_waypoints(ACTION_CHUNK_SIZE)
            if len(upcoming) > 0:
                self.trajectory_spheres = self.env.spawn_debug_trajectory(
                    upcoming, color=[1, 0.65, 0, 0.5]
                )

    def _draw_ui(self):
        """Tampilkan informasi HUD di layar simulasi."""
        state = self.vehicle.get_state()
        mode_color = [0, 1, 0] if self.mode == MODE_AUTONOMOUS else [1, 1, 0]

        info_lines = [
            ("AV SIMULATOR — Diffusion Policy Demo", [1, 1, 1], 0.95),
            ("", [1, 1, 1], 0.90),
            (f"Mode: {self.mode}", mode_color, 0.86),
            (f"Control: {self.control_mode.upper()}", [0.5, 0.8, 1], 0.82),
            (f"Action Chunking: {'ON' if self.use_action_chunking else 'OFF'}",
             [0.3, 1, 0.3] if self.use_action_chunking else [1, 0.3, 0.3], 0.78),
            (f"Recording: {'●' if self.is_recording else '○'} {len(self.current_demo)} steps",
             [1, 0, 0] if self.is_recording else [0.5, 0.5, 0.5], 0.74),
            (f"Demos saved: {len(self.demonstrations)}", [1, 1, 0], 0.70),
            (f"Replay: {'▶' if self.is_replaying else '■'}",
             [0, 1, 0] if self.is_replaying else [0.5, 0.5, 0.5], 0.66),
            (f"Steps: {self.step_count}", [1, 1, 1], 0.62),
            (f"Speed: {state['speed']:.2f} m/s", [0.5, 1, 0.5], 0.58),
            (f"Goal: {'✅ REACHED!' if self.goal_reached else '⏳ navigating...'}",
             [0, 1, 0] if self.goal_reached else [1, 1, 1], 0.54),
            ("", [1, 1, 1], 0.48),
            ("─── CONTROLS ───", [1, 1, 1], 0.44),
            ("WASD/Arrows: Drive | SPACE: Mode", [0.7, 0.7, 0.7], 0.40),
            ("R: Record Demo | P: Replay Demo", [0.7, 0.7, 0.7], 0.36),
            ("C: Action Chunking | 1/2: Ctrl Mode", [0.7, 0.7, 0.7], 0.32),
            ("Xbox: D-Pad/Sticks/Triggers", [0.7, 0.7, 0.7], 0.28),
        ]

        for text, color, y_pos in info_lines:
            rgb_color = color[:3] if len(color) > 3 else color
            try:
                p.addUserDebugText(text, [-4.5, -4.8, 0.1], rgb_color, textSize=0.4,
                                   parentObjectUniqueId=0,
                                   lifeTime=0.12)
            except:
                pass

        # ===== WAYPOINT LABELS (ADD EVERY FRAME) - SUPER VISIBLE =====
        for i, wp in enumerate(self.controller.waypoints):
            try:
                # WHITE TEXT, VERY LARGE SIZE (5.0 is huge)
                p.addUserDebugText(
                    f"WP{i}",
                    [wp[0], wp[1], 0.8],  # Higher position
                    [1, 1, 1],  # Pure white
                    textSize=5.0,  # VERY LARGE
                    lifeTime=0.15  # Per frame
                )
            except:
                pass

    def cleanup(self):
        """Cleanup resources."""
        self.env.cleanup_debug(self.trajectory_spheres)
        self.input_handler.cleanup()
        self.env.disconnect()
