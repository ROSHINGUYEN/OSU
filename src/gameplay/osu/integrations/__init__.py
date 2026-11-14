"""
Gameplay integration exports for osu! AI.
"""

from .vts_bridge import (
    ComboBreakEvent,
    ComboMilestoneEvent,
    OsuMotionBridge,
    OsuVTSBridge,
    SongCompletedEvent,
    SongFailedEvent,
    TargetHitEvent,
    osu_vts_bridge,
)

__all__ = [
    "OsuMotionBridge",
    "OsuVTSBridge",
    "osu_vts_bridge",
    "TargetHitEvent",
    "ComboMilestoneEvent",
    "ComboBreakEvent",
    "SongCompletedEvent",
    "SongFailedEvent",
]
