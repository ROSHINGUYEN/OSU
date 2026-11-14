"""
Comprehensive Unit Tests for Hit Circle and Slider Recognition & Song Sizing.
Validates:
1. HitObject & Slider CS inheritance and end_position calculation.
2. Stacking on slider tail and head with rigid curve_points shifting.
3. Catmull-Rom spline and Linear extension slider curve mathematics.
4. Dynamic per-song PlayfieldPreprocessor circle size & follow radius scaling.
5. Adaptive VisionDetector contour thresholds across varying CS.
6. High-velocity slider tracking step density (8ms sampling for v > 2.2 px/ms).
"""

import math
import numpy as np
import pytest

from src.gameplay.osu.beatmap.models import (
    BeatmapData,
    CurveType,
    DifficultySection,
    GeneralSection,
    HitCircle,
    HitObject,
    HitObjectFlag,
    Slider,
    Spinner,
)
from src.gameplay.osu.beatmap.parser import BeatmapParser
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.math.slider_curves import (
    calculate_slider_path,
    clip_points_to_length,
    sample_catmull_rom,
    sample_perfect_circle,
)
from src.gameplay.osu.perception.detector import VisionDetector
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.world.object_lifecycle import WorldObject


def test_hit_object_cs_and_end_position_attributes():
    """Verify HitObject, HitCircle, Slider, and Spinner support cs and end_position."""
    circle = HitCircle(x=100.0, y=150.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE, cs=5.5)
    assert circle.cs == 5.5
    assert circle.end_position == (100.0, 150.0)

    spinner = Spinner(x=256.0, y=192.0, time_ms=2000.0, hit_type=HitObjectFlag.SPINNER, cs=3.0, end_time_ms=3000.0)
    assert spinner.cs == 3.0
    assert spinner.end_position == (256.0, 192.0)

    # Slider with odd repeat count (tail is at last curve point)
    slider_odd = Slider(
        x=100.0,
        y=100.0,
        time_ms=4000.0,
        hit_type=HitObjectFlag.SLIDER,
        cs=4.0,
        curve_type=CurveType.LINEAR,
        curve_points=[(200.0, 100.0)],
        repeat_count=1,
        pixel_length=100.0,
        duration_ms=500.0,
        end_time_ms=4500.0,
    )
    assert slider_odd.cs == 4.0
    assert slider_odd.end_position == (200.0, 100.0)
    assert slider_odd.tail_position == (200.0, 100.0)

    # Slider with even repeat count (tail returns to head)
    slider_even = Slider(
        x=100.0,
        y=100.0,
        time_ms=5000.0,
        hit_type=HitObjectFlag.SLIDER,
        cs=4.0,
        curve_type=CurveType.LINEAR,
        curve_points=[(200.0, 100.0)],
        repeat_count=2,
        pixel_length=100.0,
        duration_ms=1000.0,
        end_time_ms=6000.0,
    )
    assert slider_even.end_position == (100.0, 100.0)


def test_stacking_on_slider_tail_and_head():
    """Verify BeatmapParser stacking recognizes notes placed on slider tails as well as heads."""
    raw_osu = """osu file format v14
[General]
Mode: 0
StackLeniency: 0.7

[Difficulty]
CircleSize: 4
ApproachRate: 9
SliderMultiplier: 1.4

[HitObjects]
100,100,1000,2,0,L|200:100,1,100,0:0|0:0,0:0:0:0:
200,100,1400,1,0,0:0:0:0:
"""
    beatmap = BeatmapParser.parse_string(raw_osu)
    slider = beatmap.hit_objects[0]
    circle_at_tail = beatmap.hit_objects[1]

    assert isinstance(slider, Slider)
    assert isinstance(circle_at_tail, HitCircle)

    # The circle was placed at (200, 100), which is the exact tail of the slider!
    assert circle_at_tail.stack_height == 1
    cs_scale = max(0.1, 0.85 - (0.07 * 4.0))  # 0.57
    expected_shift = 1 * cs_scale * -6.4
    assert pytest.approx(circle_at_tail.x, abs=1e-3) == 200.0 + expected_shift
    assert pytest.approx(circle_at_tail.y, abs=1e-3) == 100.0 + expected_shift


def test_catmull_rom_spline_slider_curve():
    """Verify Catmull-Rom spline interpolation (C) produces smooth continuous path through all points."""
    pts = [(100.0, 100.0), (150.0, 180.0), (220.0, 140.0), (300.0, 250.0)]
    spline = sample_catmull_rom(pts, pixel_length=0.0)

    # Must start near first point and end near last point
    assert pytest.approx(spline[0][0], abs=1e-2) == 100.0
    assert pytest.approx(spline[0][1], abs=1e-2) == 100.0
    assert pytest.approx(spline[-1][0], abs=1e-2) == 300.0
    assert pytest.approx(spline[-1][1], abs=1e-2) == 250.0
    assert len(spline) >= 15

    # Slider using Catmull curve enum
    slider_catmull = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.CATMULL,
        curve_points=[(150.0, 180.0), (220.0, 140.0), (300.0, 250.0)],
        repeat_count=1,
        pixel_length=250.0,
        duration_ms=600.0,
        end_time_ms=1600.0,
    )
    evaluated = calculate_slider_path(slider_catmull)
    assert len(evaluated) > 10
    assert pytest.approx(evaluated[0][0], abs=1e-2) == 100.0


def test_linear_slider_extension_when_length_exceeds_points():
    """Verify linear slider polyline extends along final direction when pixel_length > control distance."""
    # Points distance is 100.0, but pixel_length is 150.0
    pts = [(100.0, 100.0), (200.0, 100.0)]
    extended = clip_points_to_length(pts, target_len=150.0)

    assert len(extended) == 3
    assert extended[0] == (100.0, 100.0)
    assert extended[1] == (200.0, 100.0)
    assert pytest.approx(extended[2][0], abs=1e-2) == 250.0
    assert pytest.approx(extended[2][1], abs=1e-2) == 100.0


def test_playfield_preprocessor_dynamic_cs_and_radii():
    """Verify PlayfieldPreprocessor calculates radius, hitbox, and follow circle across CS values."""
    prep = PlayfieldPreprocessor(screen_width=1600, screen_height=900)
    # Scale = 1.875
    assert prep.scale == 1.875

    # CS 4.0: osu_radius = 54.4 - 4.48 * 4.0 = 36.48. Screen radius = 36.48 * 1.875 = 68.4px
    prep.set_circle_size(4.0)
    assert prep.active_circle_size == 4.0
    r_cs4 = prep.get_hit_circle_radius()
    assert pytest.approx(r_cs4, abs=1e-2) == 68.4
    assert pytest.approx(prep.get_hitbox_radius(), abs=1e-2) == 68.4
    assert pytest.approx(prep.get_slider_follow_radius(is_lazer=True), abs=1e-2) == 68.4 * 3.0
    assert pytest.approx(prep.get_slider_follow_radius(is_lazer=False), abs=1e-2) == 68.4 * 2.4

    # CS 5.5: osu_radius = 54.4 - 4.48 * 5.5 = 29.76. Screen radius = 29.76 * 1.875 = 55.8px
    prep.set_circle_size(5.5)
    r_cs55 = prep.get_hit_circle_radius()
    assert pytest.approx(r_cs55, abs=1e-2) == 55.8
    assert pytest.approx(prep.get_circle_diameter(), abs=1e-2) == 111.6


def test_vision_detector_adaptive_cs_thresholds():
    """Verify VisionDetector updates min/max radius and area filters when circle size changes."""
    prep = PlayfieldPreprocessor(screen_width=1600, screen_height=900)
    detector = VisionDetector(prep.bounds)

    # Initial state (default)
    assert detector.circle_radius is None

    # Adapt to CS 2.0 (large circle: osu_r = 45.44, screen_r = 85.2px)
    detector.update_circle_size(2.0, prep.bounds.scale)
    assert pytest.approx(detector.circle_radius, abs=1e-2) == 85.2

    # Adapt to CS 7.0 (tiny circle: osu_r = 23.04, screen_r = 43.2px)
    detector.update_circle_size(7.0, prep.bounds.scale)
    assert pytest.approx(detector.circle_radius, abs=1e-2) == 43.2


def test_high_velocity_slider_tracking_sampling_density():
    """Verify StandardBrain generates tight 8ms sampling steps for high-velocity sliders (>2.2 px/ms)."""
    prep = PlayfieldPreprocessor(screen_width=1600, screen_height=900)
    brain = StandardBrain(keys=["z", "x"], od=8.0, cs=4.0, preprocessor=prep)

    # Fast slider: 500 osu pixels in 150ms -> velocity = 3.33 px/ms (> 2.2 px/ms)
    fast_slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        cs=4.0,
        curve_type=CurveType.LINEAR,
        curve_points=[(600.0, 100.0)],
        repeat_count=1,
        pixel_length=500.0,
        duration_ms=150.0,
        end_time_ms=1150.0,
    )
    w_obj = WorldObject(id="obj_fast_slider", hit_object=fast_slider)
    intents = brain.plan_actions(
        current_time_ms=900.0,
        objects=[w_obj],
        timeline_anchor_ns=0,
    )

    slide_moves = [a for a in intents if "slide_" in a.action_id and "dwell" not in a.action_id]
    # For a 150ms slider with 8ms step, we expect at least 15-19 steps
    assert len(slide_moves) >= 15, f"Expected dense waypoints for fast slider, got {len(slide_moves)}"
