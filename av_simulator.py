"""
╔════════════════════════════════════════════════════════════════╗
║  Diffusion Policy Simulator — Main Simulator Class            ║
║  Autonomous Navigation dengan Full Pipeline Visualization     ║
║  Observation → Encoder → Diffusion → Action → Control         ║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p
import numpy as np
from collections import deque
from av_config import *
from av_input_simple import InputHandler
from av_vehicle import Racecar
from av_environment import Environment
from av_controller import PurePursuitController
from av_diffusion_pipeline import DiffusionPipeline


class DiffusionPolicySimulator:
    """Simulator autonomous navigation dengan Diffusion Policy pipeline.

    Pipeline:
    Observation → Encoder → Conditional Diffusion Model → Action → Vehicle Control
    """

    def __init__(self, env, vehicle, controller, input_handler):
        self.env = env
        self.vehicle = vehicle
        self.controller = controller
        self.input_handler = input_handler
        self.diffusion_pipeline = DiffusionPipeline()

        # State
        self.running = True
        self.use_action_chunking = True

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
        _, _, toggles = self.input_handler.get_action()

        # Proses toggles
        self._handle_toggles(toggles)

        # --- Autonomous Step ---
        self._step_autonomous()

        # --- Step Physics ---
        p.stepSimulation()

        # --- Cek goal ---
        if not self.goal_reached:
            state = self.vehicle.get_state()
            goal_pos = self.controller.waypoints[-1][:2]
            dist_to_goal = np.linalg.norm(state["position"][:2] - goal_pos)
            if dist_to_goal < WAYPOINT_THRESHOLD:
                self.goal_reached = True
                print(f"\033[92m[MISSION_COMPLETE] 🎯 GOAL TERCAPAI dalam {self.step_count} steps!\033[0m")

        self.step_count += 1

        # --- UI Text ---
        self._draw_ui()

    def _handle_toggles(self, toggles):
        """Handle toggle commands dari input handler."""
        if toggles.get("chunking"):
            self.use_action_chunking = not self.use_action_chunking
            print(f"\033[96m[INFO] Action Chunking: {'AKTIF' if self.use_action_chunking else 'NONAKTIF'}\033[0m")

    def _step_autonomous(self):
        """Autonomous step dengan full Diffusion Policy pipeline."""
        state = self.vehicle.get_state()

        # Pure Pursuit control
        throttle, steering = self.controller.compute_action(state)

        # ACTION CHUNKING
        if self.use_action_chunking:
            self.action_chunk_buffer.append([throttle, steering])
            if len(self.action_chunk_buffer) >= ACTION_CHUNK_SIZE:
                chunk = np.mean(self.action_chunk_buffer, axis=0)
                throttle, steering = chunk[0], chunk[1]

        # Apply action ke vehicle
        self.vehicle.apply_action(throttle, steering, CTRL_POSITION)

        # ===== PIPELINE VISUALIZATION =====
        # Get target waypoint
        if self.controller.current_wp_idx < len(self.controller.waypoints):
            target_wp = self.controller.waypoints[self.controller.current_wp_idx]
        else:
            target_wp = self.controller.waypoints[-1]

        # Stage 1: OBSERVATION
        obs = self.diffusion_pipeline.extract_observation(state, target_wp)

        # Stage 2: ENCODER
        feat = self.diffusion_pipeline.encode_observation()

        # Stage 3: CONDITIONAL DIFFUSION (iterative denoising untuk display)
        self.diffusion_pipeline.diffusion_history = []
        for _ in range(self.diffusion_pipeline.max_diffusion_steps):
            self.diffusion_pipeline.diffusion_denoise_iteration(throttle, steering)

        # Stage 4: ACTION SEQUENCE
        self.diffusion_pipeline.predicted_action = np.array([throttle, steering])

        # Print pipeline output
        print(self.diffusion_pipeline.get_terminal_output())

        # ===== WAYPOINT STATUS =====
        current_wp_idx = self.controller.current_wp_idx
        total_waypoints = len(self.controller.waypoints)

        if current_wp_idx < total_waypoints:
            dist_to_wp = np.linalg.norm(state["position"][:2] - self.controller.waypoints[current_wp_idx][:2])
            progress = f"{current_wp_idx}/{total_waypoints-1}"
            print(f"\033[94m[AUTONOMOUS] Heading to WP{progress} | Distance: {dist_to_wp:.2f}m | Speed: {state['speed']:.2f} m/s\033[0m")
        else:
            print(f"\033[94m[AUTONOMOUS] ALL WAYPOINTS COMPLETED!\033[0m")

        # Visualize trajectory
        self._visualize_trajectory_preview(state)

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

        info_lines = [
            ("AV SIMULATOR — Diffusion Policy", [1, 1, 1], 0.95),
            ("", [1, 1, 1], 0.90),
            ("Mode: AUTONOMOUS", [0, 1, 0], 0.86),
            (f"Control: POSITION CONTROL", [0.5, 0.8, 1], 0.82),
            (f"Action Chunking: {'ON' if self.use_action_chunking else 'OFF'}",
             [0.3, 1, 0.3] if self.use_action_chunking else [1, 0.3, 0.3], 0.78),
            (f"Steps: {self.step_count}", [1, 1, 1], 0.74),
            (f"Speed: {state['speed']:.2f} m/s", [0.5, 1, 0.5], 0.70),
            (f"Goal: {'✅ REACHED!' if self.goal_reached else '⏳ navigating...'}",
             [0, 1, 0] if self.goal_reached else [1, 1, 1], 0.66),
            ("", [1, 1, 1], 0.60),
            ("─── CONTROLS ───", [1, 1, 1], 0.56),
            ("C: Toggle Action Chunking", [0.7, 0.7, 0.7], 0.52),
            ("Q: Quit", [0.7, 0.7, 0.7], 0.48),
        ]

        for text, color, y_pos in info_lines:
            rgb_color = color[:3] if len(color) > 3 else color
            try:
                p.addUserDebugText(text, [-4.5, -4.8, 0.1], rgb_color, textSize=0.4,
                                   parentObjectUniqueId=0,
                                   lifeTime=0.12)
            except:
                pass

        # ===== WAYPOINT LABELS =====
        for i, wp in enumerate(self.controller.waypoints):
            try:
                p.addUserDebugText(
                    f"WP{i}",
                    [wp[0], wp[1], 0.8],
                    [1, 1, 1],
                    textSize=5.0,
                    lifeTime=0.15
                )
            except:
                pass

    def cleanup(self):
        """Cleanup resources."""
        self.env.cleanup_debug(self.trajectory_spheres)
        self.input_handler.cleanup()
        self.env.disconnect()
