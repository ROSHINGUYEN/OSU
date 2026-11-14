"""
Beatmap Loader and Discovery Service.
Finds, loads, and syncs .osu beatmaps with the CentralStateStore.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from src.gameplay.osu.beatmap.cache import beatmap_cache
from src.gameplay.osu.beatmap.models import BeatmapData
from src.gameplay.osu.beatmap.parser import BeatmapParser
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import BeatmapState, GameMode, state_store


class BeatmapLoader:
    """Service for discovering, parsing, and caching beatmaps."""

    @classmethod
    def load_from_file(cls, file_path: str, use_cache: bool = True) -> Tuple[BeatmapData, HitObjectTimeline]:
        """Load, parse, and index a beatmap from a file path."""
        norm_path = str(Path(file_path).resolve())
        if use_cache:
            cached = beatmap_cache.get(norm_path)
            if cached:
                return cached

        beatmap = BeatmapParser.parse_file(norm_path)
        timeline = HitObjectTimeline(beatmap.hit_objects)

        if use_cache:
            beatmap_cache.put(norm_path, beatmap, timeline)

        return beatmap, timeline

    @classmethod
    def load_from_string(cls, content: str, key: str = "in_memory") -> Tuple[BeatmapData, HitObjectTimeline]:
        """Parse beatmap directly from text content."""
        beatmap = BeatmapParser.parse_string(content, file_path=key)
        timeline = HitObjectTimeline(beatmap.hit_objects)
        beatmap_cache.put(key, beatmap, timeline)
        return beatmap, timeline

    @classmethod
    def sync_to_state_store(cls, beatmap: BeatmapData) -> BeatmapState:
        """Atomically update BeatmapState in CentralStateStore."""
        return state_store.mutate_domain(
            "beatmap",
            lambda curr: BeatmapState(
                map_id=str(beatmap.metadata.beatmap_id or beatmap.raw_path),
                title=beatmap.metadata.title,
                artist=beatmap.metadata.artist,
                creator=beatmap.metadata.creator,
                od=beatmap.difficulty.overall_difficulty,
                cs=beatmap.difficulty.circle_size,
                ar=beatmap.difficulty.approach_rate,
                hp=beatmap.difficulty.hp_drain_rate,
                total_objects=beatmap.total_objects,
                mode=getattr(beatmap, "mode", GameMode.UNKNOWN),
                is_loaded=True,
                updated_at_ns=clock.now_ns(),
            ),
            caller="beatmap_service",
        )

    @classmethod
    def scan_directory(cls, dir_path: str) -> List[str]:
        """Scan a folder recursively for .osu files."""
        found: List[str] = []
        p = Path(dir_path)
        if not p.is_dir():
            return found
        for file in p.rglob("*.osu"):
            found.append(str(file.resolve()))
        return found
