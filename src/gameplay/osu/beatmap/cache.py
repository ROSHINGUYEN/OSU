"""
Beatmap In-Memory Cache for Pre-Parsed Maps and Timelines.
"""

from typing import Dict, Optional, Tuple
from src.gameplay.osu.beatmap.models import BeatmapData
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline


class BeatmapCache:
    """Thread-safe cache holding parsed BeatmapData and indexer timelines."""

    def __init__(self, max_entries: int = 50):
        self._max_entries = max_entries
        self._cache: Dict[str, Tuple[BeatmapData, HitObjectTimeline]] = {}

    def get(self, key: str) -> Optional[Tuple[BeatmapData, HitObjectTimeline]]:
        """Retrieve cached (BeatmapData, HitObjectTimeline) by key."""
        return self._cache.get(key)

    def put(self, key: str, beatmap: BeatmapData, timeline: HitObjectTimeline) -> None:
        """Store parsed beatmap and timeline."""
        if len(self._cache) >= self._max_entries:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = (beatmap, timeline)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# Global default beatmap cache instance
beatmap_cache = BeatmapCache()
