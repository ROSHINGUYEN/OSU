"""
Feedback Evaluator for Shiori AI osu! Gameplay.
Correlates action executions with gameplay outcomes, tracks observation provenance,
and drives adaptive timing calibration and game memory updates.
"""

from enum import Enum
from typing import List, Optional

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    SessionMetrics,
    state_store,
)
from src.gameplay.osu.evaluation.metrics import GameplayMetrics
from src.gameplay.osu.timing.calibration import timing_calibrator
from src.gameplay.osu.timing.hit_windows import HitWindowModel
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState
from src.gameplay.osu.world.world_state import (
    WorldStateManager,
    world_state_manager,
)


class ObservationSource(str, Enum):
    """Provenance source for timing and hit accuracy observations."""
    OBSERVED = "observed"      # Direct memory telemetry from Tosu / Gosumemory
    ESTIMATED = "estimated"    # Optical vision detection or interpolated calculation
    SIMULATED = "simulated"    # Synthetic benchmark or test environment
    UNKNOWN = "unknown"        # Unverified source


class FeedbackEvaluator:
    """Evaluates hits and misses, updates metrics, and drives adaptive timing calibration."""

    def __init__(self, world_mgr: WorldStateManager = world_state_manager):
        self._world_mgr = world_mgr
        self.metrics = GameplayMetrics()
        self.calibrator = timing_calibrator
        self.provenance: ObservationSource = ObservationSource.UNKNOWN
        self.verified_timing_errors_ms: List[float] = []
        self._last_seen_tosu_error_count: int = 0

    def record_tosu_hit_errors(
        self,
        hit_errors: List[float],
        source: ObservationSource = ObservationSource.OBSERVED,
    ) -> None:
        """
        Ingest authoritative hit errors directly from Tosu memory client.
        Performs deduplication against cumulative hit error arrays.
        """
        if not hit_errors:
            return

        self.provenance = source

        # Handle song restart / array reset
        if len(hit_errors) < self._last_seen_tosu_error_count:
            self._last_seen_tosu_error_count = 0

        new_errors = hit_errors[self._last_seen_tosu_error_count:]
        if new_errors:
            for err in new_errors:
                self.metrics.timing_errors_ms.append(float(err))
                if len(self.metrics.timing_errors_ms) > 500:
                    self.metrics.timing_errors_ms.pop(0)
                self.verified_timing_errors_ms.append(float(err))
            self._last_seen_tosu_error_count = len(hit_errors)
            self._sync_to_store()

    def evaluate_hit(
        self,
        obj_id: str,
        delta_ms: float,
        od: float,
        mode: GameMode,
        confidence: float = 1.0,
        source: ObservationSource = ObservationSource.ESTIMATED,
    ) -> str:
        """
        Evaluates a hit timing delta, updates metrics and adaptive calibration.
        Returns result string ('300', '100', '50', 'miss').
        """
        if self.provenance == ObservationSource.UNKNOWN:
            self.provenance = source

        result = HitWindowModel.evaluate_timing_error(delta_ms, od, mode)
        self.metrics.record_result(result, delta_ms)

        # Feed error to adaptive calibration on valid hits
        if result in ("300", "100"):
            timing_calibrator.record_hit_sample(delta_ms, confidence)

        # Update world object lifecycle
        w_obj = self._world_mgr.get_object(obj_id)
        if w_obj:
            w_obj.hit_result = result
            w_obj.timing_error_ms = delta_ms
            if result != "miss":
                w_obj.transition_to(ObjectLifecycleState.HIT_CONFIRMED)
            else:
                w_obj.transition_to(ObjectLifecycleState.MISS_CONFIRMED)

        self._sync_to_store()
        return result

    def record_miss(self, obj_id: str) -> None:
        """Explicitly record a missed note."""
        self.metrics.record_result("miss")
        w_obj = self._world_mgr.get_object(obj_id)
        if w_obj:
            w_obj.hit_result = "miss"
            w_obj.transition_to(ObjectLifecycleState.MISS_CONFIRMED)
        self._sync_to_store()

    def reset(self) -> None:
        self.metrics.reset()
        self.provenance = ObservationSource.UNKNOWN
        self.verified_timing_errors_ms.clear()
        self._last_seen_tosu_error_count = 0
        self._sync_to_store()

    def _sync_to_store(self) -> None:
        state_store.mutate_domain(
            "metrics",
            lambda curr: SessionMetrics(
                hit_counts={
                    "300": self.metrics.count_300,
                    "100": self.metrics.count_100,
                    "50": self.metrics.count_50,
                    "miss": self.metrics.count_miss,
                },
                max_combo=self.metrics.max_combo,
                unstable_rate=self.metrics.unstable_rate,
                actions_submitted=curr.actions_submitted + 1,
                stale_actions_rejected=curr.stale_actions_rejected,
                updated_at_ns=clock.now_ns(),
            ),
            caller="feedback_evaluator",
        )


feedback_evaluator = FeedbackEvaluator()
