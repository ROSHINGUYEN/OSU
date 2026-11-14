"""
Pre-Input Action Intent Schema and Safety Validator.
Guarantees integrity of execution deadlines, coordinates, and key mappings before queue submission.
"""

from typing import Optional, Tuple
from src.gameplay.osu.input.action_model import ActionIntent, ActionType


class ActionValidator:
    """Validates ActionIntents before they can enter the Central Input Queue."""

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        try:
            import ctypes
            rw = ctypes.windll.user32.GetSystemMetrics(0)
            rh = ctypes.windll.user32.GetSystemMetrics(1)
            self.screen_width = max(rw, screen_width)
            self.screen_height = max(rh, screen_height)
        except Exception:
            self.screen_width = screen_width
            self.screen_height = screen_height

    def validate(self, action: ActionIntent) -> Tuple[bool, Optional[str]]:
        """
        Validates action intent against safety contracts.
        Returns (is_valid, rejection_reason).
        """
        # 1. Deadline check
        if action.execute_at_ns > action.deadline_ns:
            return False, f"Invalid timing: execute_at ({action.execute_at_ns}) > deadline ({action.deadline_ns})"

        # 2. Movement coordinate bounds check
        if action.action_type == ActionType.MOUSE_MOVE:
            if action.x is None or action.y is None:
                return False, "MOUSE_MOVE action missing (x, y) coordinates"
            if action.x < 0 or action.x > self.screen_width or action.y < 0 or action.y > self.screen_height:
                return False, f"Coordinates out of bounds: ({action.x}, {action.y})"

        # 3. Mouse button action check
        if action.action_type in [ActionType.MOUSE_DOWN, ActionType.MOUSE_UP, ActionType.MOUSE_CLICK]:
            button = getattr(action, "button", "left") or "left"
            if button not in ["left", "right", "middle"]:
                return False, f"Invalid mouse button: '{button}'"

        # 4. Key action check
        if action.action_type in [ActionType.KEY_DOWN, ActionType.KEY_UP, ActionType.KEY_TAP]:
            if not action.key:
                return False, f"Key action {action.action_type.value} missing key specification"

        return True, None


action_validator = ActionValidator()
