"""
World State Management Service.
Maintains active gameplay entities, aggregates tracking data, and coordinates with CentralStateStore.
"""

from typing import Dict, List, Optional

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import WorldState, state_store
from src.gameplay.osu.world.object_lifecycle import (
    ObjectLifecycleState,
    WorldObject,
)


class WorldStateManager:
    """Manages all active WorldObject instances during a gameplay session."""

    def __init__(self):
        self._objects: Dict[str, WorldObject] = {}
        self._current_combo: int = 0
        self._accuracy: float = 100.0
        self._health: float = 1.0

    @property
    def active_objects(self) -> Dict[str, WorldObject]:
        return self._objects

    def add_or_update(self, obj: WorldObject) -> None:
        self._objects[obj.id] = obj

    def get_object(self, obj_id: str) -> Optional[WorldObject]:
        return self._objects.get(obj_id)

    def transition_state(self, obj_id: str, new_state: ObjectLifecycleState) -> bool:
        if obj_id in self._objects:
            return self._objects[obj_id].transition_to(new_state)
        return False

    def get_by_state(self, state: ObjectLifecycleState) -> List[WorldObject]:
        return [o for o in self._objects.values() if o.lifecycle == state]

    def clear(self) -> None:
        self._objects.clear()
        self._current_combo = 0
        self._accuracy = 100.0
        self._health = 1.0

    def prune_expired(self, current_time_ms: float, max_age_ms: float = 600.0) -> int:
        """
        Removes objects that have expired or finished execution in the past.
        Keeps the active working set small (< 25 objects) to maintain sub-millisecond
        planning and serialization speed.
        """
        cutoff = current_time_ms - max_age_ms
        dead_keys = []
        for k, v in self._objects.items():
            start_t = getattr(v.hit_object, "time_ms", 0.0)
            dur = getattr(v.hit_object, "duration_ms", 0.0) or 0.0
            raw_end = getattr(v.hit_object, "end_time_ms", 0.0) or 0.0
            obj_end = raw_end if raw_end > start_t else (start_t + dur)
            if (
                obj_end < cutoff
                and v.lifecycle in (
                    ObjectLifecycleState.HIT_CONFIRMED,
                    ObjectLifecycleState.MISS_CONFIRMED,
                    ObjectLifecycleState.EXPIRED,
                    ObjectLifecycleState.ACTION_SENT,
                )
            ) or (obj_end < current_time_ms - 5000.0):
                dead_keys.append(k)
        for k in dead_keys:
            del self._objects[k]
        return len(dead_keys)

    def sync_to_central_store(self) -> WorldState:
        """Atomically persist world snapshot to CentralStateStore."""
        obj_summaries = {
            k: {
                "lifecycle": v.lifecycle.value,
                "screen_x": v.screen_x,
                "screen_y": v.screen_y,
                "time_ms": v.hit_object.time_ms,
                "confidence": v.confidence,
            }
            for k, v in self._objects.items()
        }

        return state_store.mutate_domain(
            "world",
            lambda curr: WorldState(
                objects=obj_summaries,
                current_combo=self._current_combo,
                accuracy=self._accuracy,
                health=self._health,
                state_version=curr.state_version + 1,
                updated_at_ns=clock.now_ns(),
            ),
            caller="world_service",
        )


world_state_manager = WorldStateManager()
