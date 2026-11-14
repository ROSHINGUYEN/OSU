"""
Chronological HitObject Timeline and Fast Window Lookup Engine.
Provides O(log N) binary search queries for rolling-horizon planning.
"""

import bisect
from typing import List, Optional

from src.gameplay.osu.beatmap.models import HitObject, Slider, Spinner


class HitObjectTimeline:
    """
    Timeline indexing hit objects by timestamp.
    Supports high-speed time window queries required by the Rolling-Horizon Planner.
    """

    def __init__(self, hit_objects: List[HitObject]):
        # Guarantee objects are sorted by time_ms
        self._objects: List[HitObject] = sorted(hit_objects, key=lambda o: o.time_ms)
        self._timestamps: List[float] = [o.time_ms for o in self._objects]

    @property
    def total_objects(self) -> int:
        return len(self._objects)

    @property
    def is_empty(self) -> bool:
        return len(self._objects) == 0

    @property
    def start_time_ms(self) -> float:
        return self._timestamps[0] if self._timestamps else 0.0

    @property
    def end_time_ms(self) -> float:
        if not self._objects:
            return 0.0
        last = self._objects[-1]
        if isinstance(last, (Slider, Spinner)):
            return getattr(last, "end_time_ms", last.time_ms)
        return last.time_ms

    @property
    def duration_ms(self) -> float:
        return max(0.0, self.end_time_ms - self.start_time_ms)

    def get_objects_in_window(self, start_ms: float, end_ms: float) -> List[HitObject]:
        """
        Fast binary search to retrieve all hit objects starting within [start_ms, end_ms].
        Time complexity: O(log N + K).
        """
        if not self._timestamps or start_ms > end_ms:
            return []

        idx_start = bisect.bisect_left(self._timestamps, start_ms)
        idx_end = bisect.bisect_right(self._timestamps, end_ms)

        return self._objects[idx_start:idx_end]

    def get_next_object(self, current_time_ms: float) -> Optional[HitObject]:
        """Get the earliest object that starts at or after current_time_ms."""
        if not self._timestamps:
            return None
        idx = bisect.bisect_left(self._timestamps, current_time_ms)
        if idx < len(self._objects):
            return self._objects[idx]
        return None

    def get_active_sliders_or_holds(self, current_time_ms: float) -> List[HitObject]:
        """
        Get all sliders, spinners, or mania hold notes currently active
        at current_time_ms (start_time <= current_time <= end_time).
        """
        if not self._objects:
            return []

        active: List[HitObject] = []
        # Lookback up to 10 seconds before current_time to check long sliders/holds
        lookback_ms = current_time_ms - 10000.0
        idx_start = bisect.bisect_left(self._timestamps, lookback_ms)
        idx_end = bisect.bisect_right(self._timestamps, current_time_ms)

        for obj in self._objects[idx_start:idx_end]:
            if isinstance(obj, (Slider, Spinner)):
                if obj.time_ms <= current_time_ms <= obj.end_time_ms:
                    active.append(obj)

        return active
