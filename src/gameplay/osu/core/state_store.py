"""
Domain-Separated Central State Store for Shiori AI osu! Gameplay System.
Strict domain partitioning, state versioning, and access control.
"""

import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class OperatingState(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    OBSERVING = "observing"
    PLAYING = "playing"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    RECOVERING = "recovering"
    SAFE_STOP = "safe_stop"
    FAILED = "failed"


class GameMode(str, Enum):
    STANDARD = "standard"
    UNKNOWN = "unknown"

    # Purged secondary modes permanently redirected to pure osu! Standard
    TAIKO = "standard"
    CATCH = "standard"
    MANIA = "standard"

    @classmethod
    def _missing_(cls, value):
        if str(value).lower() in ("taiko", "catch", "ctb", "mania", "piano"):
            return cls.STANDARD
        return cls.UNKNOWN


class ModeDetectionStatus(str, Enum):
    UNKNOWN = "unknown"
    DETECTING = "detecting"
    MODE_CONFIRMED = "mode_confirmed"
    MODE_CONFLICT = "mode_conflict"
    MODE_UNCERTAIN = "mode_uncertain"


class RecoveryFsmState(str, Enum):
    IDLE = "idle"
    REQUESTED = "requested"
    WAITING_FOR_DEPENDENCIES = "waiting_for_dependencies"
    RECOVERING = "recovering"
    BACKOFF = "backoff"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==========================================
# Domain State Definitions
# ==========================================

@dataclass(frozen=True)
class SystemState:
    state: OperatingState = OperatingState.INITIALIZING
    cpu_load: float = 0.0
    fps: float = 0.0
    sequence_id: int = 0
    updated_at_ns: int = 0


@dataclass(frozen=True)
class ModeState:
    mode: GameMode = GameMode.UNKNOWN
    status: ModeDetectionStatus = ModeDetectionStatus.UNKNOWN
    confidence: float = 0.0
    user_overridden: bool = False
    updated_at_ns: int = 0


@dataclass(frozen=True)
class BeatmapState:
    map_id: str = ""
    title: str = ""
    artist: str = ""
    creator: str = ""
    od: float = 8.0
    cs: float = 4.0
    ar: float = 9.0
    hp: float = 6.0
    total_objects: int = 0
    mode: GameMode = GameMode.UNKNOWN
    is_loaded: bool = False
    updated_at_ns: int = 0


@dataclass(frozen=True)
class WorldState:
    objects: Dict[str, Any] = field(default_factory=dict)
    playfield_scale: float = 1.0
    current_combo: int = 0
    health: float = 1.0
    accuracy: float = 100.0
    state_version: int = 0
    updated_at_ns: int = 0


@dataclass(frozen=True)
class TimingState:
    monotonic_time_ns: int = 0
    estimated_audio_time_ms: float = 0.0
    system_latency_ms: float = 5.0
    adaptive_offset_ms: float = 0.0
    updated_at_ns: int = 0


@dataclass(frozen=True)
class PlannerState:
    current_plan: List[Any] = field(default_factory=list)
    planning_horizon_ms: float = 1000.0
    replan_counter: int = 0
    updated_at_ns: int = 0


@dataclass(frozen=True)
class InputState:
    key_states: Dict[str, str] = field(default_factory=dict)  # "UP", "DOWN"
    cursor_pos: tuple = (0.0, 0.0)
    stuck_key_warnings: int = 0
    updated_at_ns: int = 0


@dataclass(frozen=True)
class SafetyState:
    dependency_health: Dict[str, bool] = field(default_factory=dict)
    active_leases: Dict[str, Any] = field(default_factory=dict)
    emergency_stop_triggered: bool = False
    updated_at_ns: int = 0


@dataclass(frozen=True)
class RecoveryState:
    state: RecoveryFsmState = RecoveryFsmState.IDLE
    retry_count: int = 0
    active_fault: str = ""
    updated_at_ns: int = 0


@dataclass(frozen=True)
class SessionMetrics:
    hit_counts: Dict[str, int] = field(default_factory=lambda: {"300": 0, "100": 0, "50": 0, "miss": 0})
    max_combo: int = 0
    unstable_rate: float = 0.0
    actions_submitted: int = 0
    stale_actions_rejected: int = 0
    updated_at_ns: int = 0


# ==========================================
# Central State Store
# ==========================================

class CentralStateStore:
    """
    Centralized store partitioned by domain.
    Provides atomic mutations, state versioning, and access control policies.
    """

    ALLOWED_MUTATORS: Dict[str, Set[str]] = {
        "world": {"world_service", "reconciliation_service", "system"},
        "timing": {"timing_engine", "calibration_service", "system"},
        "planner": {"planner", "gameplay_brain", "system"},
        "input": {"input_controller", "system"},
        "safety": {"safety_supervisor", "emergency_stop", "system"},
        "recovery": {"recovery_orchestrator", "system"},
        "mode": {"mode_detector", "system"},
        "beatmap": {"beatmap_service", "system"},
        "system": {"app", "safety_supervisor", "system"},
        "metrics": {"feedback_evaluator", "system"},
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._version = 0

        # Initialize all domains
        self._system = SystemState()
        self._mode = ModeState()
        self._beatmap = BeatmapState()
        self._world = WorldState()
        self._timing = TimingState()
        self._planner = PlannerState()
        self._input = InputState()
        self._safety = SafetyState()
        self._recovery = RecoveryState()
        self._metrics = SessionMetrics()

        self._listeners: Dict[str, List[Callable[[Any], None]]] = {}

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    # Domain Accessors (Thread-safe read)
    @property
    def system(self) -> SystemState:
        with self._lock:
            return self._system

    @property
    def mode(self) -> ModeState:
        with self._lock:
            return self._mode

    @property
    def beatmap(self) -> BeatmapState:
        with self._lock:
            return self._beatmap

    @property
    def world(self) -> WorldState:
        with self._lock:
            return self._world

    @property
    def timing(self) -> TimingState:
        with self._lock:
            return self._timing

    @property
    def planner(self) -> PlannerState:
        with self._lock:
            return self._planner

    @property
    def input(self) -> InputState:
        with self._lock:
            return self._input

    @property
    def safety(self) -> SafetyState:
        with self._lock:
            return self._safety

    @property
    def recovery(self) -> RecoveryState:
        with self._lock:
            return self._recovery

    @property
    def metrics(self) -> SessionMetrics:
        with self._lock:
            return self._metrics

    def subscribe(self, domain_name: str, listener: Callable[[Any], None]) -> None:
        """Subscribe to mutations of a specific domain."""
        with self._lock:
            if domain_name not in self._listeners:
                self._listeners[domain_name] = []
            if listener not in self._listeners[domain_name]:
                self._listeners[domain_name].append(listener)

    def _notify(self, domain_name: str, new_state: Any) -> None:
        listeners = self._listeners.get(domain_name, [])
        for listener in listeners:
            try:
                listener(new_state)
            except Exception:
                pass

    def mutate_domain(self, domain_name: str, mutator_fn: Callable[[Any], Any], caller: str = "system") -> Any:
        """
        Atomically update a domain's state after checking mutation permissions.
        Increments store version and notifies listeners.
        """
        domain_name = domain_name.lower()
        with self._lock:
            # Check access policy
            allowed = self.ALLOWED_MUTATORS.get(domain_name, {"system"})
            if caller != "system" and caller not in allowed:
                raise PermissionError(
                    f"Caller '{caller}' is unauthorized to mutate domain '{domain_name}'. "
                    f"Allowed callers: {allowed}"
                )

            current = getattr(self, f"_{domain_name}", None)
            if current is None:
                raise ValueError(f"Unknown state domain: '{domain_name}'")

            new_state = mutator_fn(current)
            setattr(self, f"_{domain_name}", new_state)
            self._version += 1

        self._notify(domain_name, new_state)
        return new_state


# Global state store instance
state_store = CentralStateStore()
