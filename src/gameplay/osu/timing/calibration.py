"""
Adaptive Timing Offset and Latency Calibration Engine.
Continuously measures feedback errors to eliminate audio-visual-input latency.
"""

from typing import List, Optional

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import TimingState, state_store


class AdaptiveTimingCalibrator:
    """Estimates system latency and adaptively learns timing offset."""

    def __init__(
        self,
        default_latency_ms: float = 14.5,
        learning_rate: float = 0.12,
        max_offset_ms: float = 8.0,
    ):
        self._system_latency_ms = default_latency_ms
        self._baseline_hardware_offset_ms = 0.0
        self._adaptive_offset_ms = 0.0
        self._learning_rate = learning_rate
        self._max_offset_ms = max_offset_ms
        self._sample_errors: List[float] = []
        self._last_consumed_hit_count: int = 0

    @property
    def system_latency_ms(self) -> float:
        return self._system_latency_ms

    @property
    def baseline_hardware_offset_ms(self) -> float:
        return self._baseline_hardware_offset_ms

    @property
    def adaptive_offset_ms(self) -> float:
        return self._adaptive_offset_ms

    @property
    def total_lead_time_ms(self) -> float:
        """Total composite lead time before note hit window: system + hardware baseline + adaptive offset."""
        return self._system_latency_ms + self._baseline_hardware_offset_ms + self._adaptive_offset_ms

    def set_system_latency_ms(self, latency_ms: float) -> None:
        self._system_latency_ms = max(0.5, latency_ms)
        self._sync_to_store()

    def set_baseline_hardware_offset_ms(self, offset_ms: float) -> float:
        """Sets the fixed OS/hardware compensation baseline (e.g. 12.5ms for Windows SendInput)."""
        self._baseline_hardware_offset_ms = max(0.0, offset_ms)
        self._sync_to_store()
        return self.total_lead_time_ms

    def calibrate_for_bpm(self, bpm: float, is_memory_mode: Optional[bool] = None) -> float:
        """
        Dynamically tunes system latency based on beatmap tempo:
        - Memory mode (Tosu 0ms latency audio clock):
          Fast BPM (>175): 1.0ms
          Normal BPM (<=175): 2.0ms
        - Optics / Optical capture mode:
          Fast BPM (>175): 9.0ms
          Standard BPM (<=175): 12.0ms
        """
        if is_memory_mode is None:
            try:
                from src.gameplay.osu.integrations.tosu_client import tosu_client
                is_memory_mode = tosu_client.is_connected
            except Exception:
                is_memory_mode = False

        if is_memory_mode:
            if bpm > 175.0:
                self._system_latency_ms = 1.0
            else:
                self._system_latency_ms = 2.0
        else:
            if bpm > 175.0:
                self._system_latency_ms = 9.0
            else:
                self._system_latency_ms = 12.0
        self._sync_to_store()
        return self._system_latency_ms

    def consume_hit_errors(self, hit_errors: List[float]) -> float:
        """
        Closed-loop real-time timing feedback from Tosu hitErrorArray.
        Filters out notelock/release outliers (abs(err) > 35ms),
        and applies smooth proportional correction to adaptive_offset_ms.
        Positive error = hit too late -> increase adaptive_offset_ms (dispatches earlier).
        Negative error = hit too early -> decrease adaptive_offset_ms (dispatches later).
        """
        if not hit_errors:
            return self._adaptive_offset_ms

        # Detect array reset on song transition or rewind
        if len(hit_errors) < self._last_consumed_hit_count:
            self._last_consumed_hit_count = 0

        if len(hit_errors) <= self._last_consumed_hit_count:
            return self._adaptive_offset_ms

        new_samples = hit_errors[self._last_consumed_hit_count:]
        self._last_consumed_hit_count = len(hit_errors)

        valid_errors = [float(e) for e in new_samples if abs(e) <= 35.0]
        if not valid_errors:
            return self._adaptive_offset_ms

        # Use median for robust rejection of accidental edge hits
        valid_errors.sort()
        median_err = valid_errors[len(valid_errors) // 2]

        # Precision Deadband: Sub-millisecond noise (|median_err| <= 1.0ms) is well within pure 300 window;
        # hold the converged offset rock-steady instead of decaying it back towards 0.0ms
        if abs(median_err) <= 1.0:
            return self._adaptive_offset_ms

        self._sample_errors.append(median_err)
        if len(self._sample_errors) > 50:
            self._sample_errors.pop(0)

        # Two-stage adaptive learning rate: fast initial convergence (0.28) on first 5 samples
        # to eliminate startup offsets within 2-3 notes, smoothly settling to steady state (0.10)
        if len(self._sample_errors) <= 5 and self._learning_rate == 0.12:
            eff_lr = 0.28
        elif self._learning_rate == 0.12:
            eff_lr = 0.10
        else:
            eff_lr = self._learning_rate

        adjustment = eff_lr * median_err
        self._adaptive_offset_ms += adjustment
        self._adaptive_offset_ms = max(-self._max_offset_ms, min(self._max_offset_ms, self._adaptive_offset_ms))
        self._sync_to_store()
        return self._adaptive_offset_ms

    def record_hit_sample(self, error_ms: float, confidence: float = 1.0) -> None:
        """
        Incorporate observed timing error:
        error_ms = actual_hit_time_ms - target_time_ms.
        Positive error = hit too late -> increase offset to hit earlier.
        """
        if confidence < 0.7 or abs(error_ms) > 35.0:
            return  # Discard untrusted or outlier hits

        self._sample_errors.append(error_ms)
        if len(self._sample_errors) > 50:
            self._sample_errors.pop(0)

        # Gradient update
        delta = self._learning_rate * error_ms
        self._adaptive_offset_ms += delta

        # Clamp offset
        self._adaptive_offset_ms = max(-self._max_offset_ms, min(self._max_offset_ms, self._adaptive_offset_ms))
        self._sync_to_store()

    def calculate_dispatch_time_ns(self, target_time_ms: float, timeline_anchor_ns: int) -> int:
        """
        Calculates exact monotonic nanosecond deadline to dispatch input.
        T_dispatch = Anchor + Target_time - (System_latency + Baseline_hardware + Adaptive_offset)
        """
        target_ns = timeline_anchor_ns + int(target_time_ms * 1_000_000)
        total_lead_ms = self.total_lead_time_ms
        lead_ns = int(total_lead_ms * 1_000_000)
        return target_ns - lead_ns

    def reset_offset(self, reset_baseline: bool = False) -> None:
        """Resets dynamic adaptive offset while safely preserving hardware baseline offset by default."""
        if reset_baseline:
            self._baseline_hardware_offset_ms = 0.0
        self._adaptive_offset_ms = 0.0
        self._sample_errors.clear()
        self._last_consumed_hit_count = 0
        self._sync_to_store()

    def set_adaptive_offset_ms(self, offset_ms: float) -> float:
        """Sets the adaptive offset directly (clamped to max_offset_ms)."""
        self._adaptive_offset_ms = max(-self._max_offset_ms, min(self._max_offset_ms, offset_ms))
        self._sync_to_store()
        return self.total_lead_time_ms

    def adjust_offset(self, delta_ms: float) -> float:
        """Manually nudge timing offset (positive = hit earlier, negative = hit later)."""
        self._adaptive_offset_ms += delta_ms
        self._adaptive_offset_ms = max(-self._max_offset_ms, min(self._max_offset_ms, self._adaptive_offset_ms))
        self._sync_to_store()
        return self.total_lead_time_ms

    def _sync_to_store(self) -> None:
        state_store.mutate_domain(
            "timing",
            lambda curr: TimingState(
                monotonic_time_ns=clock.now_ns(),
                estimated_audio_time_ms=curr.estimated_audio_time_ms,
                system_latency_ms=self._system_latency_ms,
                adaptive_offset_ms=self._adaptive_offset_ms,
                updated_at_ns=clock.now_ns(),
            ),
            caller="timing_engine",
        )


timing_calibrator = AdaptiveTimingCalibrator()
