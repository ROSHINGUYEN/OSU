"""
High-Performance, Fault-Tolerant .osu Beatmap Parser for All 4 Modes.
Parses timing points, calculates slider durations, and generates mode-specific objects.
"""

import bisect
import logging
from typing import List, Optional, Tuple

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
from src.gameplay.osu.core.state_store import GameMode

logger = logging.getLogger("Shiori.Osu.BeatmapParser")


class BeatmapParser:
    """Parser for .osu beatmap files with full support for Standard, Taiko, Catch, and Mania."""

    @classmethod
    def parse_file(cls, file_path: str) -> BeatmapData:
        """Parse a .osu file from disk."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return cls.parse_string(content, file_path=file_path)

    @classmethod
    def parse_string(cls, content: str, file_path: str = "") -> BeatmapData:
        """Parse .osu beatmap text content."""
        general = GeneralSection()
        metadata = MetadataSection()
        difficulty = DifficultySection()
        timing_points: List[TimingPoint] = []
        raw_hit_objects: List[str] = []

        current_section = ""

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Section header
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                continue

            # Key-Value sections
            if current_section == "General":
                cls._parse_key_value(line, general)
            elif current_section == "Metadata":
                cls._parse_key_value(line, metadata)
            elif current_section == "Difficulty":
                cls._parse_key_value(line, difficulty)
            elif current_section == "TimingPoints":
                tp = cls._parse_timing_point(line)
                if tp:
                    timing_points.append(tp)
            elif current_section == "HitObjects":
                raw_hit_objects.append(line)

        # Sort timing points chronologically, ensuring Uninherited (BPM) precedes Inherited (SV) at identical timestamps
        timing_points.sort(key=lambda t: (t.time_ms, 0 if t.uninherited else 1))

        # Pure Standard engine: All beatmaps are standardized to GameMode.STANDARD
        mode_int = general.mode
        if mode_int != 0:
            logger.info(f"Beatmap specifies non-standard mode {mode_int}. Purged to pure GameMode.STANDARD.")
        game_mode = GameMode.STANDARD

        # Parse hit objects as pure Standard objects (HitCircle, Slider, Spinner)
        hit_objects = cls._parse_hit_objects(raw_hit_objects, game_mode, difficulty, timing_points, general)

        return BeatmapData(
            mode=game_mode,
            general=general,
            metadata=metadata,
            difficulty=difficulty,
            timing_points=timing_points,
            hit_objects=hit_objects,
            raw_path=file_path,
        )

    @staticmethod
    def _parse_key_value(line: str, target_obj: object) -> None:
        """Parse 'Key: Value' line into dataclass fields."""
        if ":" not in line:
            return
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Map osu keys to snake_case field names
        mapping = {
            "AudioFilename": "audio_filename",
            "AudioLeadIn": "audio_lead_in",
            "PreviewTime": "preview_time",
            "Countdown": "countdown",
            "SampleSet": "sample_set",
            "StackLeniency": "stack_leniency",
            "Mode": "mode",
            "LetterboxInBreaks": "letterbox_in_breaks",
            "WidescreenStoryboard": "widescreen_storyboard",
            "Title": "title",
            "TitleUnicode": "title_unicode",
            "Artist": "artist",
            "ArtistUnicode": "artist_unicode",
            "Creator": "creator",
            "Version": "version",
            "Source": "source",
            "Tags": "tags",
            "BeatmapID": "beatmap_id",
            "BeatmapSetID": "beatmap_set_id",
            "HPDrainRate": "hp_drain_rate",
            "CircleSize": "circle_size",
            "OverallDifficulty": "overall_difficulty",
            "ApproachRate": "approach_rate",
            "SliderMultiplier": "slider_multiplier",
            "SliderTickRate": "slider_tick_rate",
        }

        field_name = mapping.get(key)
        if field_name and hasattr(target_obj, field_name):
            field_type = type(getattr(target_obj, field_name))
            try:
                if field_type is bool:
                    setattr(target_obj, field_name, val == "1" or val.lower() == "true")
                elif field_type is int:
                    setattr(target_obj, field_name, int(float(val)))
                elif field_type is float:
                    setattr(target_obj, field_name, float(val))
                elif field_name == "tags":
                    setattr(target_obj, field_name, val.split())
                else:
                    setattr(target_obj, field_name, val)
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to cast {key}={val} to {field_type}: {e}")

    @staticmethod
    def _parse_timing_point(line: str) -> Optional[TimingPoint]:
        """Parse TimingPoint line: time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects."""
        parts = line.split(",")
        if len(parts) < 2:
            return None
        try:
            time_ms = float(parts[0])
            beat_length = float(parts[1])
            meter = int(parts[2]) if len(parts) > 2 else 4
            sample_set = int(parts[3]) if len(parts) > 3 else 0
            sample_index = int(parts[4]) if len(parts) > 4 else 0
            volume = int(parts[5]) if len(parts) > 5 else 100
            uninherited = bool(int(parts[6])) if len(parts) > 6 else True
            effects = int(parts[7]) if len(parts) > 7 else 0

            return TimingPoint(
                time_ms=time_ms,
                beat_length=beat_length,
                meter=meter,
                sample_set=sample_set,
                sample_index=sample_index,
                volume=volume,
                uninherited=uninherited,
                effects=effects,
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing timing point line '{line}': {e}")
            return None

    @classmethod
    def _parse_hit_objects(
        cls,
        lines: List[str],
        mode: GameMode,
        difficulty: DifficultySection,
        timing_points: List[TimingPoint],
        general: Optional[GeneralSection] = None,
    ) -> List[HitObject]:
        """Parse raw hit object strings into mode-specific HitObjects."""
        objects: List[HitObject] = []
        for line in lines:
            parts = line.split(",")
            if len(parts) < 5:
                continue

            try:
                x = float(parts[0])
                y = float(parts[1])
                time_ms = float(parts[2])
                hit_type = int(parts[3])
                hit_sound = int(parts[4]) if len(parts) > 4 else 0

                # Filter out dummy/corrupt lines (e.g. 0,0,0,0,0 in debug maps)
                valid_type_mask = (
                    HitObjectFlag.CIRCLE
                    | HitObjectFlag.SLIDER
                    | HitObjectFlag.SPINNER
                )
                if (hit_type & valid_type_mask) == 0:
                    logger.debug(f"Skipping corrupt dummy hit object with invalid type {hit_type}: {line}")
                    continue

                is_new_combo = bool(hit_type & HitObjectFlag.NEW_COMBO)
                combo_offset = (hit_type >> 4) & 7

                obj = cls._parse_standard_object(
                    parts, x, y, time_ms, hit_type, hit_sound, is_new_combo, combo_offset, difficulty, timing_points
                )
                if obj:
                    objects.append(obj)

            except Exception as e:
                logger.debug(f"Skipping malformed hit object line '{line}': {e}")
                continue

        # Sort objects chronologically
        objects.sort(key=lambda o: o.time_ms)

        # Apply hit object stacking matching ppy/osu OsuBeatmapProcessor.cs
        if mode == GameMode.STANDARD and difficulty and general:
            cls._apply_stacking(objects, difficulty, general)

        return objects

    @classmethod
    def _apply_stacking(
        cls,
        objects: List[HitObject],
        difficulty: DifficultySection,
        general: GeneralSection,
    ) -> None:
        """
        Applies hit object stacking algorithm matching ppy/osu OsuBeatmapProcessor.cs.
        StackOffset = stack_height * scale * -6.4f.
        """
        if not objects:
            return
        import math
        from src.gameplay.osu.timing.hit_windows import HitWindowModel
        preempt = HitWindowModel.calculate_preempt_ms(difficulty.approach_rate)
        stack_threshold = preempt * getattr(general, "stack_leniency", 0.7)
        cs_scale = max(0.1, 0.85 - (0.07 * getattr(difficulty, "circle_size", 4.0)))

        # Step 1: Calculate stack heights using unshifted coordinates
        orig_positions = [(getattr(obj, "x", 0.0), getattr(obj, "y", 0.0)) for obj in objects]
        orig_curve_points = [
            list(getattr(obj, "curve_points", [])) if isinstance(obj, Slider) else []
            for obj in objects
        ]
        orig_end_positions = [
            obj.end_position if hasattr(obj, "end_position") else orig_positions[idx]
            for idx, obj in enumerate(objects)
        ]

        for i in range(len(objects)):
            obj_i = objects[i]
            if getattr(obj_i, "is_spinner", False):
                continue
            is_slider_i = isinstance(obj_i, Slider)
            for j in range(i + 1, len(objects)):
                obj_j = objects[j]
                if getattr(obj_j, "is_spinner", False):
                    continue
                end_time_i = getattr(obj_i, "end_time_ms", obj_i.time_ms)
                if (obj_j.time_ms - end_time_i) > stack_threshold:
                    break
                dist_head = math.hypot(orig_positions[j][0] - orig_positions[i][0], orig_positions[j][1] - orig_positions[i][1])
                dist_tail = (
                    math.hypot(orig_positions[j][0] - orig_end_positions[i][0], orig_positions[j][1] - orig_end_positions[i][1])
                    if is_slider_i else 999.0
                )
                if dist_head < 3.0 or dist_tail < 3.0:
                    obj_j.stack_height = max(getattr(obj_j, "stack_height", 0), getattr(obj_i, "stack_height", 0) + 1)

        # Step 2: Apply stack shift once per object
        for idx, obj in enumerate(objects):
            h = getattr(obj, "stack_height", 0)
            if h != 0:
                shift = h * cs_scale * (-6.4)
                obj.x = orig_positions[idx][0] + shift
                obj.y = orig_positions[idx][1] + shift
                if isinstance(obj, Slider):
                    if orig_curve_points[idx]:
                        obj.curve_points = [
                            (round(pt[0] + shift, 3), round(pt[1] + shift, 3))
                            for pt in orig_curve_points[idx]
                        ]
                    obj._cached_path = None
                    obj._cached_cum_lens = None

    @classmethod
    def _parse_standard_object(
        cls,
        parts: List[str],
        x: float,
        y: float,
        time_ms: float,
        hit_type: int,
        hit_sound: int,
        is_new_combo: bool,
        combo_offset: int,
        difficulty: DifficultySection,
        timing_points: List[TimingPoint],
    ) -> Optional[HitObject]:
        """Parse standard mode object (Circle, Slider, Spinner)."""
        cs_val = float(getattr(difficulty, "circle_size", 4.0))
        if hit_type & HitObjectFlag.SPINNER:
            end_time_ms = float(parts[5]) if len(parts) > 5 else time_ms
            return Spinner(
                x=x,
                y=y,
                time_ms=time_ms,
                hit_type=hit_type,
                hit_sound=hit_sound,
                is_new_combo=is_new_combo,
                combo_offset=combo_offset,
                cs=cs_val,
                end_time_ms=end_time_ms,
                duration_ms=max(0.0, end_time_ms - time_ms),
            )

        elif hit_type & HitObjectFlag.SLIDER:
            # Slider syntax: x,y,time,type,hitSound,curveType|curvePoints,repeat,pixelLength,edgeSounds,edgeSets,hitSample
            curve_data = parts[5].split("|") if len(parts) > 5 else ["B"]
            raw_curve_type = curve_data[0]
            curve_points: List[Tuple[float, float]] = []
            for pt in curve_data[1:]:
                if ":" in pt:
                    px, py = pt.split(":")
                    curve_points.append((float(px), float(py)))

            repeat_count = int(parts[6]) if len(parts) > 6 else 1
            pixel_length = float(parts[7]) if len(parts) > 7 else 0.0

            # Compute slider duration
            duration_ms = cls._compute_slider_duration(time_ms, pixel_length, repeat_count, difficulty, timing_points)
            end_time_ms = time_ms + duration_ms

            curve_enum = CurveType.BEZIER
            if raw_curve_type in [e.value for e in CurveType]:
                curve_enum = CurveType(raw_curve_type)

            return Slider(
                x=x,
                y=y,
                time_ms=time_ms,
                hit_type=hit_type,
                hit_sound=hit_sound,
                is_new_combo=is_new_combo,
                combo_offset=combo_offset,
                cs=cs_val,
                curve_type=curve_enum,
                curve_points=curve_points,
                repeat_count=repeat_count,
                pixel_length=pixel_length,
                duration_ms=duration_ms,
                end_time_ms=end_time_ms,
            )

        else:
            # Hit circle
            return HitCircle(
                x=x,
                y=y,
                time_ms=time_ms,
                hit_type=hit_type,
                hit_sound=hit_sound,
                is_new_combo=is_new_combo,
                combo_offset=combo_offset,
                cs=cs_val,
            )

    @staticmethod
    def _compute_slider_duration(
        time_ms: float,
        pixel_length: float,
        repeat_count: int,
        difficulty: DifficultySection,
        timing_points: List[TimingPoint],
    ) -> float:
        """Calculates slider duration in milliseconds."""
        if not timing_points or pixel_length <= 0:
            return 0.0

        # Find latest uninherited timing point (BPM) at or before time_ms
        active_uninherited: Optional[TimingPoint] = None
        active_inherited: Optional[TimingPoint] = None

        for tp in timing_points:
            if tp.time_ms > time_ms:
                break
            if tp.uninherited:
                active_uninherited = tp
                active_inherited = None  # Reset inherited on new timing section
            else:
                active_inherited = tp

        if active_uninherited is None:
            active_uninherited = next((tp for tp in timing_points if tp.uninherited), timing_points[0])

        beat_length = active_uninherited.beat_length
        sv_mult = active_inherited.sv_multiplier if active_inherited else 1.0

        # Pixels per beat = 100 * SliderMultiplier * sv_mult
        pixels_per_beat = 100.0 * difficulty.slider_multiplier * sv_mult
        if pixels_per_beat <= 0:
            return 0.0

        beats = (pixel_length * repeat_count) / pixels_per_beat
        duration_ms = beats * beat_length
        return max(0.0, duration_ms)
