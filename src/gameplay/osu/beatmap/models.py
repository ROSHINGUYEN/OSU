"""
Beatmap Domain Models for Shiori AI osu! Gameplay System.
Defines standard .osu data structures and mode-specific hit objects.
"""

from dataclasses import dataclass, field
from enum import IntFlag, Enum
from typing import Any, Dict, List, Optional, Tuple

from src.gameplay.osu.core.state_store import GameMode


class HitObjectFlag(IntFlag):
    CIRCLE = 1
    SLIDER = 2
    NEW_COMBO = 4
    SPINNER = 8
    COMBO_SKIP_1 = 16
    COMBO_SKIP_2 = 32
    COMBO_SKIP_3 = 64
    MANIA_HOLD = 128


class CurveType(str, Enum):
    LINEAR = "L"
    PERFECT = "P"
    BEZIER = "B"
    CATMULL = "C"


@dataclass
class GeneralSection:
    audio_filename: str = ""
    audio_lead_in: int = 0
    preview_time: int = -1
    countdown: int = 0
    sample_set: str = "Normal"
    stack_leniency: float = 0.7
    mode: int = 0  # 0=standard, 1=taiko, 2=catch, 3=mania
    letterbox_in_breaks: bool = False
    widescreen_storyboard: bool = False


@dataclass
class MetadataSection:
    title: str = ""
    title_unicode: str = ""
    artist: str = ""
    artist_unicode: str = ""
    creator: str = ""
    version: str = ""  # Difficulty name
    source: str = ""
    tags: List[str] = field(default_factory=list)
    beatmap_id: int = 0
    beatmap_set_id: int = 0


@dataclass
class DifficultySection:
    hp_drain_rate: float = 5.0
    circle_size: float = 4.0      # CS
    overall_difficulty: float = 8.0  # OD
    approach_rate: float = 9.0       # AR
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0


@dataclass(frozen=True)
class TimingPoint:
    time_ms: float
    beat_length: float  # >0: ms/beat; <0: slider velocity multiplier (-100 / beat_length)
    meter: int = 4
    sample_set: int = 0
    sample_index: int = 0
    volume: int = 100
    uninherited: bool = True
    effects: int = 0

    @property
    def bpm(self) -> float:
        if self.beat_length > 0:
            return 60000.0 / self.beat_length
        return 0.0

    @property
    def sv_multiplier(self) -> float:
        if self.beat_length < 0:
            return max(0.1, min(10.0, -100.0 / self.beat_length))
        return 1.0


@dataclass
class HitObject:
    x: float
    y: float
    time_ms: float
    hit_type: int
    hit_sound: int = 0
    is_new_combo: bool = False
    combo_offset: int = 0
    stack_height: int = 0
    cs: float = 4.0
    is_don: Optional[bool] = None
    is_katsu: Optional[bool] = None
    is_big: Optional[bool] = None
    is_drumroll: Optional[bool] = None
    is_denden: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.is_katsu is None:
            self.is_katsu = bool(self.hit_sound & 2 or self.hit_sound & 8)
        if self.is_don is None:
            self.is_don = not self.is_katsu
        if self.is_big is None:
            self.is_big = bool(self.hit_sound & 4)
        if self.is_drumroll is None:
            self.is_drumroll = bool(self.hit_type & HitObjectFlag.SLIDER)
        if self.is_denden is None:
            self.is_denden = bool(self.hit_type & HitObjectFlag.SPINNER)

    @property
    def end_position(self) -> Tuple[float, float]:
        """End position (x, y) on the osu! playfield. Defaults to (x, y) for hit circles and spinners."""
        return (self.x, self.y)

    @property
    def is_circle(self) -> bool:
        return bool(self.hit_type & HitObjectFlag.CIRCLE)

    @property
    def is_slider(self) -> bool:
        return bool(self.hit_type & HitObjectFlag.SLIDER)

    @property
    def is_spinner(self) -> bool:
        return bool(self.hit_type & HitObjectFlag.SPINNER)

    @property
    def is_hold(self) -> bool:
        return bool(self.hit_type & HitObjectFlag.MANIA_HOLD)


@dataclass
class HitCircle(HitObject):
    pass


@dataclass
class Slider(HitObject):
    curve_type: CurveType = CurveType.BEZIER
    curve_points: List[Tuple[float, float]] = field(default_factory=list)
    repeat_count: int = 1
    pixel_length: float = 0.0
    duration_ms: float = 0.0
    end_time_ms: float = 0.0
    edge_sounds: List[int] = field(default_factory=list)
    _cached_path: Optional[List[Tuple[float, float]]] = field(default=None, repr=False)
    _cached_cum_lens: Optional[List[float]] = field(default=None, repr=False)

    @property
    def end_position(self) -> Tuple[float, float]:
        """Calculates authentic tail position of the slider considering repeat counts."""
        span_count = max(1, self.repeat_count)
        if span_count % 2 == 0:
            return (self.x, self.y)
        if self._cached_path is None:
            try:
                from src.gameplay.osu.math.slider_curves import calculate_slider_path
                self._cached_path = calculate_slider_path(self)
            except Exception:
                pass
        if self._cached_path and len(self._cached_path) > 0:
            return self._cached_path[-1]
        if self.curve_points:
            return self.curve_points[-1]
        return (self.x, self.y)

    @property
    def tail_position(self) -> Tuple[float, float]:
        """Alias for end_position."""
        return self.end_position


@dataclass
class Spinner(HitObject):
    end_time_ms: float = 0.0
    duration_ms: float = 0.0


@dataclass
class BeatmapData:
    mode: GameMode
    general: GeneralSection = field(default_factory=GeneralSection)
    metadata: MetadataSection = field(default_factory=MetadataSection)
    difficulty: DifficultySection = field(default_factory=DifficultySection)
    timing_points: List[TimingPoint] = field(default_factory=list)
    hit_objects: List[HitObject] = field(default_factory=list)
    raw_path: str = ""

    @property
    def total_objects(self) -> int:
        return len(self.hit_objects)

    @property
    def start_time_ms(self) -> float:
        if not self.hit_objects:
            return 0.0
        return self.hit_objects[0].time_ms

    @property
    def end_time_ms(self) -> float:
        if not self.hit_objects:
            return 0.0
        last = self.hit_objects[-1]
        if isinstance(last, (Slider, Spinner)):
            return getattr(last, "end_time_ms", last.time_ms)
        return last.time_ms
