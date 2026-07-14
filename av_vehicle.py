"""
╔════════════════════════════════════════════════════════════════╗
║  Racecar Vehicle Model — PyBullet                             ║
║  Wrapper untuk model racecar URDF dari pybullet_data           ║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p
import pybullet_data
import numpy as np
import math
from av_config import *


class Racecar:
    """Wrapper untuk model racecar URDF dari pybullet_data."""

    def __init__(self, start_pos=[0, 0, 0.1], start_orn=[0, 0, 0, 1]):
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.body_id = None
        self.joint_indices = {}
        self.joint_name_to_index = {}
        self.current_speed = 0.0  # Track speed for smooth acceleration
        self._load()

    def _load(self):
        """Load racecar dari pybullet_data."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.body_id = p.loadURDF(
            "racecar/racecar.urdf",
            self.start_pos,
            self.start_orn,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Mapping nama joint → index
        for i in range(p.getNumJoints(self.body_id)):
            info = p.getJointInfo(self.body_id, i)
            name = info[1].decode("utf-8")
            self.joint_name_to_index[name] = i

        # Indeks joint penting
        self.joint_indices = {
            "left_rear_wheel": self.joint_name_to_index["left_rear_wheel_joint"],
            "right_rear_wheel": self.joint_name_to_index["right_rear_wheel_joint"],
            "left_front_wheel": self.joint_name_to_index["left_front_wheel_joint"],
            "right_front_wheel": self.joint_name_to_index["right_front_wheel_joint"],
            "left_steering": self.joint_name_to_index["left_steering_hinge_joint"],
            "right_steering": self.joint_name_to_index["right_steering_hinge_joint"],
        }

        # Reset ke posisi awal
        self.reset()

    def reset(self):
        """Reset posisi mobil ke start."""
        p.resetBasePositionAndOrientation(self.body_id, self.start_pos, self.start_orn)
        for name, idx in self.joint_indices.items():
            p.resetJointState(self.body_id, idx, 0, 0)
        self.current_speed = 0.0

    def get_state(self):
        """Ambil state mobil: posisi, orientasi, kecepatan."""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        vel, ang_vel = p.getBaseVelocity(self.body_id)
        _, _, yaw = p.getEulerFromQuaternion(orn)
        return {
            "position": np.array(pos),
            "orientation": np.array(orn),
            "yaw": yaw,
            "velocity": np.array(vel),
            "angular_velocity": np.array(ang_vel),
            "speed": np.linalg.norm(vel[:2]),
        }

    def apply_action(self, throttle, steering, control_mode=CTRL_POSITION):
        """Terapkan aksi ke kendaraan dengan dua control mode yang berbeda.

        Args:
            throttle: -1..1 (maju/mundur)
            steering: -1..1 (kiri/kanan)
            control_mode: 'position' (smooth ramp) atau 'velocity' (instant/jerky)
        """
        steer_angle = steering * MAX_STEER
        target_speed = throttle * MAX_SPEED

        # Debug output - always print for visibility
        print(f"[VEHICLE] throttle={throttle:.2f}, steering={steering:.2f}, speed={target_speed:.2f}, steer_angle={steer_angle:.2f}")

        # --- Steering (front wheels) ---
        p.setJointMotorControl2(
            self.body_id,
            self.joint_indices["left_steering"],
            p.POSITION_CONTROL,
            targetPosition=steer_angle,
            force=100,
        )
        p.setJointMotorControl2(
            self.body_id,
            self.joint_indices["right_steering"],
            p.POSITION_CONTROL,
            targetPosition=steer_angle,
            force=100,
        )

        # --- Drive (rear wheels) + Steering (rotation) ---
        # Get current vehicle state
        current_pos, current_orn = p.getBasePositionAndOrientation(self.body_id)
        _, _, current_yaw = p.getEulerFromQuaternion(current_orn)

        # CONTROL MODE DIFFERENCES - VERY OBVIOUS
        if control_mode == CTRL_POSITION:
            # MODE 1: POSITION CONTROL - SMOOTH RAMP (2 second ramp up)
            # Very gradual acceleration - noticeable smooth curve
            acceleration = 0.3  # m/s per frame (much slower for obvious smooth ramp)
            if self.current_speed < target_speed:
                self.current_speed = min(self.current_speed + acceleration, target_speed)
            elif self.current_speed > target_speed:
                self.current_speed = max(self.current_speed - acceleration, target_speed)
            else:
                self.current_speed = target_speed

            actual_speed = self.current_speed
            print(f"[MODE1-SMOOTH] ramp speed → {actual_speed:.2f} m/s (target: {target_speed:.2f})")

        else:
            # MODE 2: VELOCITY CONTROL - INSTANT (direct jump to max speed)
            # No smoothing - jerky/instant response
            self.current_speed = target_speed
            actual_speed = target_speed
            print(f"[MODE2-INSTANT] jump to {actual_speed:.2f} m/s (instant!)")

        # Calculate linear velocity in world frame based on yaw
        vx = actual_speed * math.cos(current_yaw)
        vy = actual_speed * math.sin(current_yaw)
        vz = 0  # Keep on ground

        # Calculate angular velocity for steering (rotation around Z axis)
        wheel_base = 0.33  # Typical racecar wheel base (~33cm)

        if abs(actual_speed) > 0.1:  # Only rotate when moving
            angular_z = (actual_speed / wheel_base) * math.tan(steer_angle) if abs(steer_angle) > 0.05 else 0
        else:
            angular_z = 0  # Don't rotate when stationary

        # Apply both linear and angular velocity
        p.resetBaseVelocity(
            self.body_id,
            linearVelocity=[vx, vy, vz],
            angularVelocity=[0, 0, angular_z]  # Only Z rotation (yaw)
        )

        # Set wheel angular velocities for visual effect (wheels spinning)
        wheel_spin_rate = actual_speed * 20 if actual_speed != 0 else 0

        p.resetJointState(
            self.body_id,
            self.joint_indices["left_rear_wheel"],
            targetValue=0,
            targetVelocity=wheel_spin_rate
        )
        p.resetJointState(
            self.body_id,
            self.joint_indices["right_rear_wheel"],
            targetValue=0,
            targetVelocity=wheel_spin_rate
        )

    def get_camera_image(self, width=320, height=240):
        """Ambil gambar dari kamera onboard (seperti input visual di paper)."""
        state = self.get_state()
        cam_pos = state["position"] + np.array([0.45, 0, 0.1])
        target_pos = cam_pos + np.array(
            [
                2.0 * math.cos(state["yaw"]),
                2.0 * math.sin(state["yaw"]),
                0,
            ]
        )

        view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, width / height, 0.02, 10)

        _, _, rgb, depth, _ = p.getCameraImage(
            width,
            height,
            view_matrix,
            proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return rgb, depth
