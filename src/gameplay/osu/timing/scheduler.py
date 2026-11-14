"""
Microsecond-Accurate Deadline Scheduler for Rhythm Actions.
"""

from typing import Any, Callable, Tuple
from src.gameplay.osu.core.clock import clock

Tuple_Result = Tuple[Any, int]


class DeadlineScheduler:
    """Executes callables precisely at a scheduled monotonic deadline."""

    @staticmethod
    def wait_until(deadline_ns: int, busy_threshold_ns: int = 2_000_000) -> int:
        """Blocks until deadline_ns, returning overshoot in nanoseconds."""
        return clock.sleep_until_ns(deadline_ns, busy_wait_threshold_ns=busy_threshold_ns)

    @classmethod
    def execute_at(cls, deadline_ns: int, action_fn: Callable[[], Any]) -> Tuple_Result:
        """Wait until deadline_ns, then immediately invoke action_fn()."""
        overshoot = cls.wait_until(deadline_ns)
        result = action_fn()
        return result, overshoot

