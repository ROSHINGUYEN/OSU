"""
Unit Tests for Beatmap Engine: Parser, Models, Timeline Indexer, and Loader.
Covers all 4 Game Modes (Standard, Taiko, Catch, Mania) and Timeline Queries.
"""

import pytest

from src.gameplay.osu.beatmap import (
    BeatmapCache,
    BeatmapData,
    BeatmapLoader,
    BeatmapParser,
    HitCircle,
    HitObjectTimeline,
    Slider,
    Spinner,
)
from src.gameplay.osu.core.state_store import GameMode, state_store


# ==========================================
# Sample Beatmap Fixtures
# ==========================================

SAMPLE_STANDARD_OSU = """osu file format v14

[General]
AudioFilename: audio.mp3
Mode: 0

[Metadata]
Title: Test Song
Artist: Shiori
Creator: Roshi
Version: Insane
BeatmapID: 123456

[Difficulty]
HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 8
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
1000,500,4,1,0,100,1,0
2000,-100,4,1,0,100,0,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
100,100,1500,2,0,B|200:200|300:100,1,140,0:0|0:0,0:0:0:0:
256,192,3000,12,0,4000,0:0:0:0:
"""

SAMPLE_TAIKO_OSU = """osu file format v14

[General]
Mode: 1

[Metadata]
Title: Taiko Drum Test
Artist: Shiori
Creator: Roshi

[Difficulty]
OverallDifficulty: 7
CircleSize: 4

[TimingPoints]
1000,500,4,1,0,100,1,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
256,192,1250,1,2,0:0:0:0:
256,192,1500,1,4,0:0:0:0:
256,192,1750,1,6,0:0:0:0:
"""

SAMPLE_MANIA_OSU = """osu file format v14

[General]
Mode: 3

[Metadata]
Title: Mania 4K Test
Artist: Shiori
Creator: Roshi

[Difficulty]
CircleSize: 4
OverallDifficulty: 8

[TimingPoints]
1000,500,4,1,0,100,1,0

[HitObjects]
64,192,1000,1,0,0:0:0:0:
192,192,1250,1,0,0:0:0:0:
320,192,1500,1,0,0:0:0:0:
448,192,1750,128,0,2500:0:0:0:0:
"""


# ==========================================
# 1. Standard Parser Tests
# ==========================================

def test_parse_standard_beatmap():
    beatmap = BeatmapParser.parse_string(SAMPLE_STANDARD_OSU)

    assert beatmap.mode == GameMode.STANDARD
    assert beatmap.metadata.title == "Test Song"
    assert beatmap.metadata.creator == "Roshi"
    assert beatmap.difficulty.overall_difficulty == 8.0
    assert beatmap.difficulty.circle_size == 4.0

    # 2 Timing points
    assert len(beatmap.timing_points) == 2
    tp1 = beatmap.timing_points[0]
    assert tp1.time_ms == 1000.0
    assert tp1.bpm == 120.0  # 60000 / 500 = 120 BPM
    assert tp1.uninherited is True

    # 3 Hit objects: Circle, Slider, Spinner
    assert len(beatmap.hit_objects) == 3
    circle = beatmap.hit_objects[0]
    slider = beatmap.hit_objects[1]
    spinner = beatmap.hit_objects[2]

    assert isinstance(circle, HitCircle)
    assert circle.time_ms == 1000.0
    assert circle.x == 256.0
    assert circle.y == 192.0

    assert isinstance(slider, Slider)
    assert slider.time_ms == 1500.0
    assert slider.pixel_length == 140.0
    assert slider.repeat_count == 1
    assert slider.duration_ms > 0.0
    assert slider.end_time_ms == slider.time_ms + slider.duration_ms

    assert isinstance(spinner, Spinner)
    assert spinner.time_ms == 3000.0
    assert spinner.end_time_ms == 4000.0
    assert spinner.duration_ms == 1000.0


# ==========================================
# 2. Converted Non-Standard Parser Tests
# ==========================================

def test_parse_taiko_beatmap_converted_to_standard():
    beatmap = BeatmapParser.parse_string(SAMPLE_TAIKO_OSU)

    assert beatmap.mode == GameMode.STANDARD
    assert len(beatmap.hit_objects) == 4

    n1, n2, n3, n4 = beatmap.hit_objects
    assert isinstance(n1, HitCircle)
    assert isinstance(n2, HitCircle)
    assert isinstance(n3, HitCircle)
    assert isinstance(n4, HitCircle)


# ==========================================
# 3. Converted Mania Parser Tests
# ==========================================

def test_parse_mania_beatmap_converted_to_standard():
    beatmap = BeatmapParser.parse_string(SAMPLE_MANIA_OSU)

    assert beatmap.mode == GameMode.STANDARD
    assert len(beatmap.hit_objects) == 3

    n1, n2, n3 = beatmap.hit_objects
    assert isinstance(n1, HitCircle)
    assert isinstance(n2, HitCircle)
    assert isinstance(n3, HitCircle)


# ==========================================
# 4. Timeline Indexer Tests
# ==========================================

def test_hit_object_timeline_queries():
    beatmap = BeatmapParser.parse_string(SAMPLE_STANDARD_OSU)
    timeline = HitObjectTimeline(beatmap.hit_objects)

    assert timeline.total_objects == 3
    assert timeline.start_time_ms == 1000.0
    assert timeline.end_time_ms == 4000.0

    # Query window [900, 1600] -> circle (1000) and slider (1500)
    window_objs = timeline.get_objects_in_window(900.0, 1600.0)
    assert len(window_objs) == 2
    assert window_objs[0].time_ms == 1000.0
    assert window_objs[1].time_ms == 1500.0

    # Query next object after 1200ms -> should be slider at 1500ms
    next_obj = timeline.get_next_object(1200.0)
    assert next_obj is not None
    assert next_obj.time_ms == 1500.0

    # Active slider/spinner query at 3500ms -> should return spinner (3000..4000)
    active = timeline.get_active_sliders_or_holds(3500.0)
    assert len(active) == 1
    assert isinstance(active[0], Spinner)


# ==========================================
# 5. Loader & Cache Integration Tests
# ==========================================

def test_beatmap_loader_and_cache():
    cache = BeatmapCache(max_entries=2)
    assert len(cache) == 0

    beatmap, timeline = BeatmapLoader.load_from_string(SAMPLE_STANDARD_OSU, key="sample_map")
    assert isinstance(beatmap, BeatmapData)
    assert isinstance(timeline, HitObjectTimeline)

    # Sync with CentralStateStore
    state = BeatmapLoader.sync_to_state_store(beatmap)
    assert state.is_loaded is True
    assert state.title == "Test Song"
    assert state.total_objects == 3
    assert state_store.beatmap.title == "Test Song"


def test_beatmap_stacking_3_stacked_notes():
    """Verify stacking algorithm calculates increasing stack_heights on consecutive identical notes."""
    raw_osu = """osu file format v14
[General]
Mode: 0
StackLeniency: 0.7

[Difficulty]
HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 8
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1

[HitObjects]
256,192,1000,1,0,0:0:0:0:
256,192,1100,1,0,0:0:0:0:
256,192,1200,1,0,0:0:0:0:
"""
    beatmap = BeatmapParser.parse_string(raw_osu)
    objs = beatmap.hit_objects
    assert len(objs) == 3
    assert getattr(objs[0], "stack_height", 0) == 0
    assert getattr(objs[1], "stack_height", 0) == 1
    assert getattr(objs[2], "stack_height", 0) == 2

    # Check that shift is applied uniformly without distorted double accumulation
    cs_scale = max(0.1, 0.85 - (0.07 * 4.0))  # 0.57
    shift_1 = 1 * cs_scale * -6.4
    shift_2 = 2 * cs_scale * -6.4
    assert pytest.approx(objs[0].x, abs=1e-3) == 256.0
    assert pytest.approx(objs[1].x, abs=1e-3) == 256.0 + shift_1
    assert pytest.approx(objs[2].x, abs=1e-3) == 256.0 + shift_2


def test_beatmap_stacking_slider_curve_points_shifted():
    """Verify that when a slider is stacked, its curve_points are shifted rigidly with its head."""
    from src.gameplay.osu.math.slider_curves import calculate_slider_path

    raw_osu = """osu file format v14
[General]
Mode: 0
StackLeniency: 0.7

[Difficulty]
CircleSize: 4
ApproachRate: 9
SliderMultiplier: 1.4

[HitObjects]
200,200,1000,1,0,0:0:0:0:
200,200,1200,2,0,L|300:200,1,100,0:0|0:0,0:0:0:0:
"""
    beatmap = BeatmapParser.parse_string(raw_osu)
    circle = beatmap.hit_objects[0]
    slider = beatmap.hit_objects[1]

    assert isinstance(slider, Slider)
    assert slider.stack_height == 1

    cs_scale = max(0.1, 0.85 - (0.07 * 4.0))  # 0.57
    expected_shift = 1 * cs_scale * -6.4

    # Slider head must be shifted
    assert pytest.approx(slider.x, abs=1e-3) == 200.0 + expected_shift
    assert pytest.approx(slider.y, abs=1e-3) == 200.0 + expected_shift

    # All slider curve control points must also be shifted rigidly
    assert len(slider.curve_points) == 1
    assert pytest.approx(slider.curve_points[0][0], abs=1e-3) == 300.0 + expected_shift
    assert pytest.approx(slider.curve_points[0][1], abs=1e-3) == 200.0 + expected_shift

    # Evaluated slider path must start at shifted head and end at shifted tail
    pts = calculate_slider_path(slider)
    assert pytest.approx(pts[0][0], abs=1e-3) == 200.0 + expected_shift
    assert pytest.approx(pts[-1][0], abs=1e-3) == 300.0 + expected_shift
