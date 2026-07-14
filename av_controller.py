"""
╔════════════════════════════════════════════════════════════════╗
║  Pure Pursuit Controller — Waypoint Navigation                ║
║  Implementasi Pure Pursuit untuk autonomous path following     ║
╚════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import math
from av_config import *


class PurePursuitController:
    """Pure Pursuit controller untuk path following.

    Kaitannya dengan paper Diffusion Policy:
      - Setiap waypoint adalah "action chunk" → prediksi posisi N langkah ke depan
      - Lookahead distance analog dengan prediction horizon di paper
      - Trajectory smoothing analog dengan denoising process
    """

    def __init__(self, waypoints, lookahead=1.0, speed=1.5):
        self.waypoints = waypoints
        self.lookahead = lookahead
        self.speed = speed
        self.current_wp_idx = 0
        self.trajectory_history = []

    def reset(self):
        """Reset ke waypoint pertama."""
        self.current_wp_idx = 0
        self.trajectory_history = []

    def compute_action(self, vehicle_state):
        """Hitung throttle & steering berdasarkan target waypoint.

        Paper: ini analog dengan compute loss / gradient untuk mencapai target.
        """
        pos = vehicle_state["position"][:2]
        yaw = vehicle_state["yaw"]

        # Cari waypoint target
        target = self._find_target_waypoint(pos)

        if target is None:
            return 0.0, 0.0  # Sudah sampai semua waypoint

        # Hitung error
        target_local = self._world_to_local(pos, yaw, target)
        cross_track_error = target_local[1]  # lateral error

        # Pure pursuit steering
        lookahead_dist = np.linalg.norm(target - pos)
        if lookahead_dist > 0.01:
            steering = math.atan2(2.0 * cross_track_error, lookahead_dist)
            steering = np.clip(steering / MAX_STEER, -1.0, 1.0)
        else:
            steering = 0.0

        # Throttle
        throttle = self.speed / MAX_SPEED

        # Kurangi kecepatan jika belok tajam
        throttle *= max(0.3, 1.0 - abs(steering) * 0.5)

        # Simpan ke trajectory history
        self.trajectory_history.append((pos.copy(), target.copy()))

        return throttle, steering

    def _find_target_waypoint(self, pos):
        """Cari waypoint terdekat yang belum dicapai."""
        if self.current_wp_idx >= len(self.waypoints):
            return None

        # Cek apakah waypoint saat ini sudah tercapai
        dist_to_current = np.linalg.norm(self.waypoints[self.current_wp_idx][:2] - pos)
        if dist_to_current < WAYPOINT_THRESHOLD:
            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.waypoints):
                return None

        # Target waypoint saat ini
        return self.waypoints[self.current_wp_idx][:2]

    def _world_to_local(self, pos, yaw, target):
        """Transformasi world → local frame."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        local_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        local_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return np.array([local_x, local_y])

    def get_upcoming_waypoints(self, n=ACTION_CHUNK_SIZE):
        """Ambil N waypoint ke depan (action chunking dari paper).

        Paper Diffusion Policy memprediksi N langkah aksi sekaligus
        dalam satu chunk, bukan 1 langkah per 1 langkah.
        """
        remaining = []
        for i in range(self.current_wp_idx, min(self.current_wp_idx + n, len(self.waypoints))):
            remaining.append(self.waypoints[i][:2])
        # Jika kurang dari n, duplicate waypoint terakhir
        while len(remaining) < n and remaining:
            remaining.append(remaining[-1])
        return np.array(remaining)
