"""
Action Intent and Input Command Models for Shiori AI osu! Gameplay.
Defines four-tier state tracking and action contracts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.gameplay.osu.core.state_store import GameMode


class ActionType(str, Enum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    MOUSE_CLICK = "mouse_click"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    KEY_TAP = "key_tap"


class CursorOwnerState(str, Enum):
    IDLE = "idle"
    MOVING_TO_TARGET = "moving_to_target"
    CIRCLE_TARGET = "circle_target"
    SLIDER_FOLLOW = "slider_follow"
    SPINNER_CONTROL = "spinner_control"
    RECOVERY = "recovery"


class InputStateStatus(str, Enum):
    INTENDED = "intended"
    SUBMISSION = "submission"
    VERIFIED = "verified"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass
class ActionIntent:
    """Action candidate scheduled for OS input execution."""
    action_id: str
    action_type: ActionType
    execute_at_ns: int
    deadline_ns: int
    session_id: str = "default_session"
    mode: GameMode = GameMode.STANDARD
    target_id: str = ""
    created_at_ns: int = 0
    priority: int = 5                  # 1=Highest/Emergency -> 10=Lowest
    x: Optional[float] = None          # Screen pixel X
    y: Optional[float] = None          # Screen pixel Y
    key: Optional[str] = None          # Character ("z", "x", "d", "f", etc.)
    button: Optional[str] = None     # Mouse button ("left", "right", "middle")
    key_code: Optional[int] = None     # Virtual key / scan code
    state_version: int = 0
    confidence: float = 1.0
    status: InputStateStatus = InputStateStatus.INTENDED
    cursor_owner: CursorOwnerState = CursorOwnerState.IDLE
    metadata: Dict[str, Any] = field(default_factory=dict)
    target_time_ms: float = 0.0  # Intended gameplay (hit) time in ms; used for timing-error math

    def is_stale(self, current_ns: int) -> bool:
        """Returns True if current time has passed action deadline."""
        return current_ns > self.deadline_ns
