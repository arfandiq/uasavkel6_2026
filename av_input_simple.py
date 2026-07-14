"""
╔════════════════════════════════════════════════════════════════╗
║  Input Handler — WASD ONLY (Simple & Clear)                   ║
║  Direct keyboard input — no arrow keys, no complexity          ║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p


class InputHandler:
    """Ultra simple input handler - WASD only."""

    def __init__(self, input_mode=None):
        self.prev_keys = {}
        print("[INPUT] ✅ WASD Keyboard Input Ready")
        print("[INPUT]    W = Forward | S = Backward")
        print("[INPUT]    A = Left   | D = Right")

    def get_action(self):
        """Get input dari keyboard. WASD ONLY."""
        throttle = 0.0
        steering = 0.0
        toggles = {}

        try:
            keys = p.getKeyboardEvents()
        except:
            keys = {}

        if not keys:
            return 0.0, 0.0, {}

        # ===== WASD MOVEMENT =====
        if ord('w') in keys:
            throttle = 1.0  # Forward
            print("[INPUT] W pressed → Forward")
        elif ord('s') in keys:
            throttle = -1.0  # Backward
            print("[INPUT] S pressed → Backward")

        if ord('a') in keys:
            steering = 1.0  # Left
            print("[INPUT] A pressed → Left")
        elif ord('d') in keys:
            steering = -1.0  # Right
            print("[INPUT] D pressed → Right")

        # ===== TOGGLE CONTROLS =====
        # Space - mode toggle
        if ord(' ') in keys and not self.prev_keys.get('space', False):
            toggles['space'] = True
            print("[INPUT] SPACE pressed → Toggle Mode")
        self.prev_keys['space'] = ord(' ') in keys

        # R - record
        if (ord('r') in keys or ord('R') in keys) and not self.prev_keys.get('r', False):
            toggles['record'] = True
            print("[INPUT] R pressed → Record Demo")
        self.prev_keys['r'] = (ord('r') in keys or ord('R') in keys)

        # P - replay
        if (ord('p') in keys or ord('P') in keys) and not self.prev_keys.get('p', False):
            toggles['replay'] = True
            print("[INPUT] P pressed → Replay Demo")
        self.prev_keys['p'] = (ord('p') in keys or ord('P') in keys)

        # C - chunking
        if (ord('c') in keys or ord('C') in keys) and not self.prev_keys.get('c', False):
            toggles['chunking'] = True
            print("[INPUT] C pressed → Toggle Action Chunking")
        self.prev_keys['c'] = (ord('c') in keys or ord('C') in keys)

        # 1 - position control
        if ord('1') in keys and not self.prev_keys.get('1', False):
            toggles['pos_ctrl'] = True
            print("[INPUT] 1 pressed → Position Control")
        self.prev_keys['1'] = ord('1') in keys

        # 2 - velocity control
        if ord('2') in keys and not self.prev_keys.get('2', False):
            toggles['vel_ctrl'] = True
            print("[INPUT] 2 pressed → Velocity Control")
        self.prev_keys['2'] = ord('2') in keys

        # Q - quit
        if ord('q') in keys or ord('Q') in keys:
            print("[INPUT] Q pressed → Quit")

        return throttle, steering, toggles

    def cleanup(self):
        """Cleanup."""
        pass

