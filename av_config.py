"""
╔════════════════════════════════════════════════════════════════╗
║  Configuration untuk AV Simulation — Diffusion Policy         ║
║  Terdapat semua konstanta dan parameter global                ║
╚════════════════════════════════════════════════════════════════╝
"""

# ============================================================================
# PHYSICS & SIMULATION
# ============================================================================
SIMULATION_STEP = 1 / 240.0  # 240 Hz
MAX_SPEED = 20.0          # m/s (reduced for stable simulation)
MAX_STEER = 0.8              # rad (~46 deg)
WAYPOINT_THRESHOLD = 0.3     # m

# ============================================================================
# ACTION CHUNKING (Paper: Diffusion Policy)
# ============================================================================
ACTION_CHUNK_SIZE = 8  # Jumlah langkah dalam satu action chunk

# ============================================================================
# MODE KONTROL
# ============================================================================
MODE_MANUAL = "MANUAL"
MODE_AUTONOMOUS = "AUTONOMOUS"

CTRL_POSITION = "position"  # Position control (paper: lebih stabil)
CTRL_VELOCITY = "velocity"  # Velocity control

# ============================================================================
# INPUT DEVICE
# ============================================================================
INPUT_KEYBOARD = "keyboard"
INPUT_XBOX = "xbox"
INPUT_BOTH = "both"  # Detect otomatis

# Xbox Controller Constants
XBOX_DEADZONE = 0.15
XBOX_TRIGGER_THRESHOLD = 0.1

# ============================================================================
# ENVIRONMENT
# ============================================================================
ARENA_SIZE = 10.0  # 10m x 10m
NUM_OBSTACLES = 4
NUM_WAYPOINTS = 8

# ============================================================================
# UI/DEBUG
# ============================================================================
ENABLE_GUI = True
ENABLE_DEBUG_TRAJECTORY = True
MAX_SIMULATION_STEPS = 5000

# ============================================================================
# GPU/RENDERING
# ============================================================================
USE_GPU_RENDERING = True
USE_OPENGL2 = True

# ============================================================================
# FILE PATHS
# ============================================================================
DEMO_OUTPUT_DIR = "./demo_data"
LOG_OUTPUT_DIR = "./logs"
