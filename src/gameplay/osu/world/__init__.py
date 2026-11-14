"""
World State Domain for Shiori AI osu! Gameplay.
Exports ObjectLifecycleState, WorldObject, WorldStateManager, and ReconciliationService.
"""

from src.gameplay.osu.world.object_lifecycle import (
    ObjectLifecycleState,
    WorldObject,
)
from src.gameplay.osu.world.reconciliation import (
    ReconciliationMode,
    ReconciliationService,
    reconciliation_service,
)
from src.gameplay.osu.world.world_state import (
    WorldStateManager,
    world_state_manager,
)

__all__ = [
    "ObjectLifecycleState",
    "WorldObject",
    "WorldStateManager",
    "world_state_manager",
    "ReconciliationMode",
    "ReconciliationService",
    "reconciliation_service",
]
