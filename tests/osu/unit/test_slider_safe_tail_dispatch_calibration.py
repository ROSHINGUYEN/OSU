"""
Unit tests verifying slider safe tail dispatch time calibration,
lead-time invariance, and adaptive timing calibrator noise deadband.
"""

import pytest
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.beatmap.models import Slider, HitCircle, CurveType
from src.gameplay.osu.world.object_lifecycle import WorldObject
from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator, timing_calibrator
from src.gameplay.osu.input.action_model import ActionType


def test_slider_key_down_and_up_lead_time_invariance():
    """
    Slider KEY_DOWN and KEY_UP must both use calculate_dispatch_time_ns.
    The key hold duration must equal exactly duration_ms + 12.0ms invariant of latency and adaptive offset.
    """
    brain = StandardBrain(keys=["z", "x"])
    slider = Slider(
        x=256.0, y=192.0, time_ms=1000.0, hit_type=2,
        curve_type=CurveType.BEZIER, curve_points=[(300.0, 200.0)],
        repeat_count=1, pixel_length=100.0, duration_ms=250.0, end_time_ms=1250.0,
    )
    w_obj = WorldObject(id="s1", hit_object=slider)
    anchor = 1_000_000_000

    # 1. Nominal calibration (adaptive_offset_ms = 0.0)
    timing_calibrator.reset_offset()
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_obj], timeline_anchor_ns=anchor)

    down_intent = next(i for i in intents if i.action_id == "act_s1_down")
    up_intent = next(i for i in intents if i.action_id == "act_s1_up")

    expected_down_ns = timing_calibrator.calculate_dispatch_time_ns(1000.0, anchor)
    expected_up_ns = timing_calibrator.calculate_dispatch_time_ns(1262.0, anchor)

    assert down_intent.execute_at_ns == expected_down_ns
    assert up_intent.execute_at_ns == expected_up_ns
    # Hold duration = 262.0ms exactly
    hold_duration_ns = up_intent.execute_at_ns - down_intent.execute_at_ns
    assert hold_duration_ns == int(262.0 * 1_000_000)

    # 2. Large adaptive offset shift (+6.0ms)
    timing_calibrator.set_adaptive_offset_ms(6.0)
    intents_shifted = brain.plan_actions(current_time_ms=900.0, objects=[w_obj], timeline_anchor_ns=anchor)

    down_shifted = next(i for i in intents_shifted if i.action_id == "act_s1_down")
    up_shifted = next(i for i in intents_shifted if i.action_id == "act_s1_up")

    assert down_shifted.execute_at_ns == timing_calibrator.calculate_dispatch_time_ns(1000.0, anchor)
    assert up_shifted.execute_at_ns == timing_calibrator.calculate_dispatch_time_ns(1262.0, anchor)
    # Crucial assertion: hold duration MUST remain exactly 262.0ms!
    hold_duration_shifted_ns = up_shifted.execute_at_ns - down_shifted.execute_at_ns
    assert hold_duration_shifted_ns == int(262.0 * 1_000_000)

    # Clean up
    timing_calibrator.reset_offset()


def test_slider_followed_by_circle_key_not_blocked():
    """
    When a slider is followed by a circle, slider key release must be synchronized
    so the circle is not blocked by a lingering slider key.
    """
    brain = StandardBrain(keys=["z", "x"])
    slider = Slider(
        x=200.0, y=200.0, time_ms=1000.0, hit_type=2,
        curve_type=CurveType.BEZIER, curve_points=[(250.0, 200.0)],
        repeat_count=1, pixel_length=80.0, duration_ms=150.0, end_time_ms=1150.0,
    )
    circle = HitCircle(x=350.0, y=200.0, time_ms=1200.0, hit_type=1)

    w_s = WorldObject(id="s1", hit_object=slider)
    w_c = WorldObject(id="c1", hit_object=circle)
    anchor = 1_000_000_000

    timing_calibrator.reset_offset()
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_s, w_c], timeline_anchor_ns=anchor)

    s1_down = next(i for i in intents if i.action_id == "act_s1_down")
    s1_up = next(i for i in intents if i.action_id == "act_s1_up")
    c1_down = next(i for i in intents if i.action_id == "act_c1_down" or i.action_id == "act_c1")

    # s1 safe tail: 1150.0 + 12.0 = 1162.0ms
    # c1 hit time: 1200.0ms
    # s1_up must execute strictly BEFORE c1_down
    assert s1_up.execute_at_ns < c1_down.execute_at_ns


def test_calibrator_sub_millisecond_noise_deadband_and_array_reset():
    """
    Verify that noise within sub-millisecond precision (|err| <= 1.0ms) does not accumulate drift,
    and array length drop resets consumption pointer.
    """
    cal = AdaptiveTimingCalibrator(default_latency_ms=14.5, learning_rate=0.12, max_offset_ms=8.0)
    assert cal.adaptive_offset_ms == 0.0

    # 1. Sub-millisecond noise (+0.5ms): within deadband, offset stays 0.0
    cal.consume_hit_errors([0.5, 0.6, 0.4])
    assert cal.adaptive_offset_ms == 0.0

    # 2. Significant real drift (+5.0ms): outside deadband, smoothly adjusts
    cal.consume_hit_errors([0.5, 0.6, 0.4, 5.0, 5.0, 5.0])
    assert cal.adaptive_offset_ms > 0.0

    # 3. Array reset simulation (new song starts: hit_errors goes from 6 elements to 2 elements)
    cal.consume_hit_errors([-6.0, -6.0])
    # Must have accepted new samples instead of ignoring due to len < 6
    assert cal._last_consumed_hit_count == 2
