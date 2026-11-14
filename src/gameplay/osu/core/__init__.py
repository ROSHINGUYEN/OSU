"""
Shiori AI osu! Gameplay Core Foundation.
Exports Clock, EventBus, DataChannels, StateStore, and Configuration models.
"""

from src.gameplay.osu.core.clock import HighResolutionClock, clock
from src.gameplay.osu.core.config import (
    CaptureConfig,
    InputConfig,
    OsuGameplayConfig,
    PlanningConfig,
    SafetyConfig,
    TimingConfig,
    default_config,
)
from src.gameplay.osu.core.data_channels import (
    DataChannel,
    frame_channel,
    telemetry_channel,
)
from src.gameplay.osu.core.event_bus import Event, EventBus, event_bus
from src.gameplay.osu.core.state_store import (
    BeatmapState,
    CentralStateStore,
    GameMode,
    InputState,
    ModeDetectionStatus,
    ModeState,
    OperatingState,
    PlannerState,
    RecoveryFsmState,
    RecoveryState,
    SafetyState,
    SessionMetrics,
    SystemState,
    TimingState,
    WorldState,
    state_store,
)

__all__ = [
    # Clock
    "HighResolutionClock",
    "clock",
    # Event Bus
    "Event",
    "EventBus",
    "event_bus",
    # Data Channels
    "DataChannel",
    "frame_channel",
    "telemetry_channel",
    # State Store
    "CentralStateStore",
    "state_store",
    "OperatingState",
    "GameMode",
    "ModeDetectionStatus",
    "RecoveryFsmState",
    "SystemState",
    "ModeState",
    "BeatmapState",
    "WorldState",
    "TimingState",
    "PlannerState",
    "InputState",
    "SafetyState",
    "RecoveryState",
    "SessionMetrics",
    # Config
    "OsuGameplayConfig",
    "CaptureConfig",
    "TimingConfig",
    "InputConfig",
    "SafetyConfig",
    "PlanningConfig",
    "default_config",
]
