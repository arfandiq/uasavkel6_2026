"""
╔════════════════════════════════════════════════════════════════╗
║  Diffusion Policy Pipeline Visualization (ADAPTED/SIMPLIFIED)  ║
║  Observation → Encoder → Action Generation (from observations) ║
║  NOT a neural network; heuristic-based for educational demo    ║
╚════════════════════════════════════════════════════════════════╝

NOTE: This is an EDUCATIONAL ADAPTATION of Diffusion Policy concepts.
In the real paper, the diffusion model is a neural network trained on
demonstrations. Here we use heuristic rules that directly map encoded
observations to actions - intuitive but simplified.

Full pipeline stages:
1. OBSERVATION: Extract raw vehicle state
2. ENCODER: Feature extraction/dimensionality reduction
3. DIFFUSION: Iterative refinement (action generation from observations)
4. ACTION: Generated action from diffusion process
5. CONTROL: Apply to vehicle
"""

import numpy as np
from av_config import *


class DiffusionPipeline:
    """
    SIMPLIFIED Diffusion Policy pipeline untuk demonstrasi educational.

    Konsep yang ditunjukkan:
    - Observation extraction (state representation)
    - Encoding (feature extraction dari raw state)
    - Action generation dari encoded observations (bukan waypoint-based)
    - Iterative refinement via diffusion-inspired denoising
    """

    def __init__(self):
        self.current_observation = None
        self.encoded_features = None
        self.diffusion_steps = 0
        self.max_diffusion_steps = 10  # Number of refinement iterations
        self.predicted_action = None
        self.diffusion_history = []  # Track action refinement across iterations
        self.frame_counter = 0  # Counter untuk output frequency control
        self.output_frequency = 1  # Print terminal output every frame

    def extract_observation(self, vehicle_state, target_waypoint):
        """
        STAGE 1: OBSERVATION
        Extract raw state representation dari vehicle dan environment.

        Ini adalah "visual input" dalam paper - semua info yang tersedia
        untuk policy untuk membuat keputusan.
        """
        pos = vehicle_state["position"][:2]
        vel = vehicle_state["velocity"][:2]
        yaw = vehicle_state["yaw"]
        speed = vehicle_state["speed"]

        # Compute heading error to waypoint
        dx = target_waypoint[0] - pos[0]
        dy = target_waypoint[1] - pos[1]
        heading_error = np.arctan2(dy, dx) - yaw
        dist_to_target = np.sqrt(dx**2 + dy**2)

        # Normalize heading error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # Observation vector: [pos_x, pos_y, vel_x, vel_y, yaw, speed, heading_error, dist_to_target]
        self.current_observation = np.array([
            pos[0], pos[1],
            vel[0], vel[1],
            yaw,
            speed,
            heading_error,
            dist_to_target
        ])

        return self.current_observation

    def encode_observation(self):
        """
        STAGE 2: ENCODER
        Extract features dari raw observation.

        Dalam paper, encoder adalah bagian dari neural network yang
        mengkonversi visual/state input menjadi latent representation.

        Di sini: simplified feature extraction dengan normalisasi.
        """
        if self.current_observation is None:
            return None

        obs = self.current_observation

        # Feature extraction: normalize dan extract meaningful features
        # Features: [normalized_heading_error, normalized_distance, normalized_speed, lateral_velocity]
        heading_error_norm = np.clip(obs[6] / np.pi, -1, 1)  # heading error normalized
        dist_norm = np.clip(obs[7] / 5.0, 0, 1)              # distance normalized (max 5m)
        speed_norm = np.clip(obs[5] / MAX_SPEED, 0, 1)       # speed normalized
        lateral_vel = np.clip(obs[3] / MAX_SPEED, -1, 1)     # lateral velocity

        self.encoded_features = np.array([
            heading_error_norm,
            dist_norm,
            speed_norm,
            lateral_vel
        ])

        return self.encoded_features

    def generate_action_from_observation(self):
        """
        STAGE 3: DIFFUSION MODEL (ACTION GENERATION)
        Generate action DIRECTLY dari encoded observation.

        BALANCED FIX: Moderate steering + throttle for smooth navigation without wheelie
        """
        if self.encoded_features is None:
            return 0.35, 0.0

        feat = self.encoded_features

        heading_error_norm = feat[0]      # -1 to 1
        dist_norm = feat[1]               # 0 to 1
        speed_norm = feat[2]              # 0 to 1

        # STEERING: Restore to working coefficient (0.5) for proper turning
        steering = np.clip(heading_error_norm * 0.5, -1, 1)

        # THROTTLE: VERY CONSERVATIVE to prevent wheelie (0.25-0.3 range, not 0.35!)
        throttle = 0.25

        if abs(steering) > 0.8:
            throttle = 0.15
        elif abs(steering) > 0.6:
            throttle = 0.2

        throttle = np.clip(throttle, 0.15, 0.3)  # REDUCED from 0.2-0.4 to 0.15-0.3

        return throttle, steering

    def diffusion_denoise_iteration(self, throttle, steering):
        """
        STAGE 3b: DIFFUSION DENOISING ITERATION

        Simulasi iterative refinement dari diffusion process.
        Setiap iteration: add noise + gradient step untuk memperbaiki action.
        """
        if len(self.diffusion_history) == 0:
            # Initialize: base action generated dari observation
            current_action = np.array([throttle, steering])
            self.diffusion_history.append(current_action.copy())
            return current_action

        # Get previous action
        current_action = self.diffusion_history[-1].copy()

        # Refinement step: smooth out the action
        # In paper: diffusion model gradually denoises from random noise to data distribution
        # Here: we gradually smooth the action to improve consistency

        noise = np.random.normal(0, 0.02, 2)  # Small random noise
        target = np.array([throttle, steering])

        # Move toward target with damping (smoother trajectory)
        alpha = 0.05  # Refinement rate
        refined_action = current_action + alpha * (target - current_action) + noise

        # Clip to valid range
        refined_action[0] = np.clip(refined_action[0], -1, 1)  # throttle
        refined_action[1] = np.clip(refined_action[1], -1, 1)  # steering

        self.diffusion_history.append(refined_action.copy())

        return refined_action

    def get_pipeline_status(self):
        """Return current status untuk terminal display."""
        status = {
            "observation": self.current_observation,
            "encoded_features": self.encoded_features,
            "diffusion_steps": len(self.diffusion_history),
            "current_action": self.predicted_action,
            "diffusion_history": self.diffusion_history
        }
        return status

    def get_terminal_output(self):
        """Format terminal output dengan frequency control — hanya setiap N frames.

        Terminal output menunjukkan 5 stages dari PPT slide 7:
        1. [OBSERVATION] — Raw state extraction
        2. [ENCODER] — Feature encoding
        3. [CONDITIONAL_DIFFUSION] — Action generation via iterative denoising
        4. [ACTION_SEQUENCE] — Predicted actions
        5. [VEHICLE_CONTROL] — Applied control (di av_vehicle.py)
        """
        self.frame_counter += 1

        # Hanya tampilkan output setiap N frames untuk mengurangi spam
        if self.frame_counter % self.output_frequency != 0:
            return ""

        if self.current_observation is None:
            return ""

        obs = self.current_observation
        feat = self.encoded_features if self.encoded_features is not None else np.zeros(4)
        action = self.predicted_action if self.predicted_action is not None else np.zeros(2)
        steps = len(self.diffusion_history)

        # Separator line untuk clarity antara frames
        separator = f"\033[90m{'═' * 130}\033[0m"

        # Terminal output color-coded untuk setiap PPT pipeline stage:
        # BRIGHT CYAN (96m) — OBSERVATION (Stage 1)
        # BRIGHT YELLOW (93m) — ENCODER (Stage 2)
        # BRIGHT BLUE (94m) — CONDITIONAL_DIFFUSION (Stage 3)
        # RED (91m) — ACTION_SEQUENCE (Stage 4)

        output = (
            f"\n{separator}\n"
            f"\033[96m[OBSERVATION] pos=({obs[0]:.2f},{obs[1]:.2f}) vel=({obs[2]:.2f},{obs[3]:.2f}) "
            f"yaw={obs[4]:.2f} heading_err={obs[6]:.2f} dist_to_target={obs[7]:.2f}\033[0m\n"
            f"\033[93m[ENCODER] normalized=[{feat[0]:.2f},{feat[1]:.2f},{feat[2]:.2f},{feat[3]:.2f}] → "
            f"4D features extracted\033[0m\n"
            f"\033[94m[CONDITIONAL_DIFFUSION] iter {steps}/{self.max_diffusion_steps} denoising: "
            f"action_seq=[({action[0]:.2f},{action[1]:.2f}),...]  (Stochastic Langevin Dynamics)\033[0m\n"
            f"\033[91m[ACTION_SEQUENCE] predicted=[steering:{action[1]:.2f}, throttle:{action[0]:.2f}, brake:0.00]\033[0m\n"
            f"{separator}\n"
        )

        return output

    def reset(self):
        """Reset pipeline state untuk next inference cycle."""
        self.current_observation = None
        self.encoded_features = None
        self.diffusion_history = []
        self.predicted_action = None
