"""
High-Precision Monotonic Clock & Deadline Scheduler Primitive.
Utilizes time.perf_counter_ns() for sub-microsecond resolution on Windows.
"""

import time
from typing import Optional


class HighResolutionClock:
    """
    High-precision monotonic clock for real-time rhythm game timing.
    Provides nanosecond accuracy and hybrid sleep/spin-wait for deadline scheduling.
    """

    def __init__(self, default_offset_ns: int = 0):
        self._offset_ns: int = default_offset_ns
        self._start_time_ns: int = time.perf_counter_ns()

    @property
    def offset_ns(self) -> int:
        return self._offset_ns

    @property
    def offset_ms(self) -> float:
        return self._offset_ns / 1_000_000.0

    def set_offset_ns(self, offset_ns: int) -> None:
        """Set or update calibrated timing offset in nanoseconds."""
        self._offset_ns = offset_ns

    def adjust_offset_ns(self, delta_ns: int) -> None:
        """Incrementally adjust the offset (e.g. by adaptive learning)."""
        self._offset_ns += delta_ns

    def now_ns(self) -> int:
        """Returns current monotonic time in nanoseconds, including calibrated offset."""
        return time.perf_counter_ns() + self._offset_ns

    def now_ms(self) -> float:
        """Returns current monotonic time in milliseconds."""
        return self.now_ns() / 1_000_000.0

    def now_s(self) -> float:
        """Returns current monotonic time in seconds."""
        return self.now_ns() / 1_000_000_000.0

    def raw_now_ns(self) -> int:
        """Returns raw monotonic counter without offset."""
        return time.perf_counter_ns()

    def elapsed_since_ns(self, start_ns: int) -> int:
        """Nanoseconds elapsed since start_ns."""
        return self.now_ns() - start_ns

    def elapsed_since_ms(self, start_ns: int) -> float:
        """Milliseconds elapsed since start_ns."""
        return (self.now_ns() - start_ns) / 1_000_000.0

    def sleep_until_ns(self, target_ns: int, busy_wait_threshold_ns: int = 2_000_000) -> int:
        """
        Precise hybrid sleep until target_ns:
        - If remaining time > busy_wait_threshold_ns (default 2ms), sleep to yield CPU.
        - Once within busy_wait_threshold_ns, busy-spin with perf_counter_ns for sub-0.1ms precision.
        Returns the overshoot in nanoseconds (actual_ns - target_ns).
        """
        while True:
            current = self.now_ns()
            remaining = target_ns - current
            if remaining <= 0:
                return -remaining  # Overshoot in ns

            # If more than busy_wait_threshold_ns, coarse sleep for 75% of remaining time
            if remaining > busy_wait_threshold_ns:
                sleep_duration_s = (remaining - busy_wait_threshold_ns) / 1_000_000_000.0
                time.sleep(max(0.0, sleep_duration_s))
            else:
                # Busy wait spinning
                while self.now_ns() < target_ns:
                    pass
                return self.now_ns() - target_ns

    def sleep_ms(self, duration_ms: float) -> int:
        """Sleep for duration_ms with hybrid precision."""
        target_ns = self.now_ns() + int(duration_ms * 1_000_000)
        return self.sleep_until_ns(target_ns)


# Global default clock instance
clock = HighResolutionClock()
