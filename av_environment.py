"""
╔════════════════════════════════════════════════════════════════╗
║  Environment Builder — Using Public PyBullet Assets            ║
║  Simplified arena dengan plane.urdf dari pybullet_data         ║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p
import pybullet_data
import numpy as np
from av_config import *


class Environment:
    """Membangun lingkungan simulasi dengan public PyBullet assets."""

    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        self.obstacles = []
        self.waypoints = []
        self.goal_id = None
        self.physics_client = None
        self._setup()

    def _setup(self):
        """Setup physics client dan scene."""
        # Koneksi GUI/DIRECT
        if self.use_gui:
            self.physics_client = p.connect(p.GUI, options="--opengl2")
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(SIMULATION_STEP)

        # Physics engine setup
        try:
            p.setPhysicsEngineParameter(
                numSubSteps=1,
                fixedTimeStep=SIMULATION_STEP,
                numSolverIterations=4,
                constraintSolverType=p.CONSTRAINT_SOLVER_LCP_PGS
            )
            print("[PHYSICS] ✅ Physics engine optimized")
        except Exception as e:
            print(f"[PHYSICS] ⚠️  Physics setup note: {e}")

        # Load plane.urdf (public PyBullet asset)
        p.loadURDF("plane.urdf")
        print("[ENV] ✅ Loaded plane.urdf from pybullet_data")

        # Add grid visualization
        self._add_grid_visualization()
        print("[ENV] ✅ Added grid visualization")

        # Lighting
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        # Build simple scene with minimal obstacles
        self._build_simple_obstacles()
        self._build_waypoints()
        self._build_goal_marker()

    def _build_simple_obstacles(self):
        """Create minimal obstacles untuk demo."""
        # No obstacles - clean arena for demo
        print("[ENV] ✅ No obstacles - clean arena")

    def _add_grid_visualization(self):
        """Add grid lines to visualize the arena (call once, lines persist)."""
        grid_size = 6.0  # Grid range
        grid_spacing = 1.0
        grid_height = 0.01
        line_lifetime = 0  # 0 = persist forever

        # Horizontal lines (along Y axis)
        x = -grid_size
        while x <= grid_size:
            from_xyz = [x, -grid_size, grid_height]
            to_xyz = [x, grid_size, grid_height]
            p.addUserDebugLine(from_xyz, to_xyz, [0.3, 0.3, 0.3], lineWidth=1, lifeTime=line_lifetime)
            x += grid_spacing

        # Vertical lines (along X axis)
        y = -grid_size
        while y <= grid_size:
            from_xyz = [-grid_size, y, grid_height]
            to_xyz = [grid_size, y, grid_height]
            p.addUserDebugLine(from_xyz, to_xyz, [0.3, 0.3, 0.3], lineWidth=1, lifeTime=line_lifetime)
            y += grid_spacing

    def _build_waypoints(self):
        """Define waypoints untuk autonomous navigation."""
        # 8 waypoints dalam lingkaran sederhana
        self.waypoints = [
            np.array([3.0, 0.0, 0.0]),
            np.array([2.1, 2.1, 0.0]),
            np.array([0.0, 3.0, 0.0]),
            np.array([-2.1, 2.1, 0.0]),
            np.array([-3.0, 0.0, 0.0]),
            np.array([-2.1, -2.1, 0.0]),
            np.array([0.0, -3.0, 0.0]),
            np.array([2.1, -2.1, 0.0]),
        ]
        print(f"[ENV] ✅ Created {len(self.waypoints)} waypoints")

    def _build_goal_marker(self):
        """Marker visual untuk goal waypoint."""
        goal_pos = self.waypoints[-1]
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.3, length=0.05,
                                  rgbaColor=[0, 1, 0, 0.6])
        self.goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                         basePosition=[goal_pos[0], goal_pos[1], 0.025])
        print("[ENV] ✅ Created goal marker")

    def spawn_debug_trajectory(self, trajectory, color=[1, 0, 0, 0.5]):
        """Visualisasi trajectory untuk action chunking."""
        sphere_ids = []
        for i, pt in enumerate(trajectory):
            r = max(0.03, 0.07 - 0.005 * i)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
            sid = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                    basePosition=[pt[0], pt[1], 0.05])
            sphere_ids.append(sid)
        return sphere_ids

    def cleanup_debug(self, ids):
        """Hapus debug visual objects."""
        for bid in ids:
            try:
                p.removeBody(bid)
            except:
                pass

    def disconnect(self):
        """Disconnect dari physics server."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)

