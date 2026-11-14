"""
Object Lifecycle State Machine for Shiori AI osu! Gameplay.
Tracks every hit object through 11 distinct operational states.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, Optional, Set

from src.gameplay.osu.beatmap.models import HitObject

logger = logging.getLogger("Shiori.Osu.ObjectLifecycle")


class ObjectLifecycleState(str, Enum):
    PREDICTED = "predicted"
    VISIBLE = "visible"
    CONFIRMED = "confirmed"
    APPROACHING = "approaching"
    ACTION_SCHEDULED = "action_scheduled"
    ACTION_SENT = "action_sent"
    RESULT_PENDING = "result_pending"
    HIT_CONFIRMED = "hit_confirmed"
    MISS_CONFIRMED = "miss_confirmed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


# Valid lifecycle transitions. Expiration (EXPIRED) and cancellation (CANCELLED)
# are permitted from any non-terminal state; all other transitions follow the
# normal gameplay progression. Terminal states (HIT_CONFIRMED, MISS_CONFIRMED,
# EXPIRED, CANCELLED) have no outgoing transitions.
_VALID_TRANSITIONS: Dict[ObjectLifecycleState, Set[ObjectLifecycleState]] = {
    ObjectLifecycleState.PREDICTED: {
        ObjectLifecycleState.VISIBLE,
        ObjectLifecycleState.CONFIRMED,
        ObjectLifecycleState.APPROACHING,
        ObjectLifecycleState.ACTION_SCHEDULED,
    },
    ObjectLifecycleState.VISIBLE: {
        ObjectLifecycleState.CONFIRMED,
        ObjectLifecycleState.APPROACHING,
        ObjectLifecycleState.ACTION_SCHEDULED,
    },
    ObjectLifecycleState.CONFIRMED: {
        ObjectLifecycleState.APPROACHING,
        ObjectLifecycleState.ACTION_SCHEDULED,
    },
    ObjectLifecycleState.APPROACHING: {
        ObjectLifecycleState.ACTION_SCHEDULED,
    },
    ObjectLifecycleState.ACTION_SCHEDULED: {
        ObjectLifecycleState.ACTION_SENT,
        ObjectLifecycleState.RESULT_PENDING,
        ObjectLifecycleState.HIT_CONFIRMED,
        ObjectLifecycleState.MISS_CONFIRMED,
    },
    ObjectLifecycleState.ACTION_SENT: {
        ObjectLifecycleState.RESULT_PENDING,
        ObjectLifecycleState.HIT_CONFIRMED,
        ObjectLifecycleState.MISS_CONFIRMED,
    },
    ObjectLifecycleState.RESULT_PENDING: {
        ObjectLifecycleState.HIT_CONFIRMED,
        ObjectLifecycleState.MISS_CONFIRMED,
    },
    ObjectLifecycleState.HIT_CONFIRMED: set(),
    ObjectLifecycleState.MISS_CONFIRMED: set(),
}

_TERMINAL_STATES = frozenset(
    {
        ObjectLifecycleState.HIT_CONFIRMED,
        ObjectLifecycleState.MISS_CONFIRMED,
        ObjectLifecycleState.EXPIRED,
        ObjectLifecycleState.CANCELLED,
    }
)


@dataclass
class WorldObject:
    """Wraps a HitObject with real-time lifecycle tracking and reconciliation data."""
    id: str
    hit_object: HitObject
    lifecycle: ObjectLifecycleState = ObjectLifecycleState.PREDICTED
    screen_x: float = 0.0
    screen_y: float = 0.0
    confidence: float = 1.0
    last_observed_ns: int = 0
    action_sent_at_ns: Optional[int] = None
    hit_result: Optional[str] = None  # "300", "100", "50", "miss"
    timing_error_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def transition_to(self, new_state: ObjectLifecycleState) -> bool:
        """Advance lifecycle state through a validated transition.

        Returns True if the transition was applied, False if it is invalid and
        was rejected. Expiration and cancellation are permitted from any
        non-terminal state; terminal states cannot transition further.
        """
        current = self.lifecycle
        if current == new_state:
            return True
        if current in _TERMINAL_STATES:
            logger.warning(
                "Rejected lifecycle transition for object %s: terminal state %s -> %s",
                self.id, current.value, new_state.value,
            )
            return False
        if new_state in (ObjectLifecycleState.EXPIRED, ObjectLifecycleState.CANCELLED):
            self.lifecycle = new_state
            return True
        allowed = _VALID_TRANSITIONS.get(current, set())
        if new_state not in allowed:
            logger.warning(
                "Rejected invalid lifecycle transition for object %s: %s -> %s",
                self.id, current.value, new_state.value,
            )
            return False
        self.lifecycle = new_state
        return True
