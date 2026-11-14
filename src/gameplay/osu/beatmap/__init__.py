"""
Beatmap Engine Package for Shiori AI osu! Gameplay.
Exports Models, Parser, Timeline Indexer, Cache, and Loader.
"""

from src.gameplay.osu.beatmap.cache import BeatmapCache, beatmap_cache
from src.gameplay.osu.beatmap.loader import BeatmapLoader
from src.gameplay.osu.beatmap.models import (
    BeatmapData,
    CurveType,
    DifficultySection,
    GeneralSection,
    HitCircle,
    HitObject,
    HitObjectFlag,
    MetadataSection,
    Slider,
    Spinner,
    TimingPoint,
)
from src.gameplay.osu.beatmap.parser import BeatmapParser
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline

__all__ = [
    # Models
    "BeatmapData",
    "HitObject",
    "HitCircle",
    "Slider",
    "Spinner",
    "TimingPoint",
    "GeneralSection",
    "MetadataSection",
    "DifficultySection",
    "HitObjectFlag",
    "CurveType",
    # Parser & Timeline
    "BeatmapParser",
    "HitObjectTimeline",
    # Loader & Cache
    "BeatmapLoader",
    "BeatmapCache",
    "beatmap_cache",
]
