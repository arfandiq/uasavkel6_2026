"""
╔════════════════════════════════════════════════════════════════╗
║  Input Handler — Keyboard Controls                            ║
║  Minimal controls untuk autonomous navigation demo            ║
╚════════════════════════════════════════════════════════════════╝
"""

import pybullet as p


class InputHandler:
    """Input handler untuk keyboard controls."""

    def __init__(self, input_mode=None):
        self.prev_keys = {}
        print("[INPUT] Keyboard Input Ready")

    def get_action(self):
        """Get input dari keyboard."""
        throttle = 0.0
        steering = 0.0
        toggles = {}

        try:
            keys = p.getKeyboardEvents()
        except:
            keys = {}

        if not keys:
            return 0.0, 0.0, {}

        # C - toggle action chunking
        if (ord('c') in keys or ord('C') in keys) and not self.prev_keys.get('c', False):
            toggles['chunking'] = True
        self.prev_keys['c'] = (ord('c') in keys or ord('C') in keys)

        # Q - quit
        if ord('q') in keys or ord('Q') in keys:
            toggles['quit'] = True

        return throttle, steering, toggles

    def cleanup(self):
        """Cleanup."""
        pass
