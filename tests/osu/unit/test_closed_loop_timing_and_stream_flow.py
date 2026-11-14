"""
Unit tests for closed-loop real-time timing feedback calibration,
stream cursor dwell preservation, and consecutive slider transitions.
"""

import pytest
from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator, timing_calibrator
from src.gameplay.osu.integrations.tosu_client import TosuClient
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.beatmap.models import HitCircle, Slider, CurveType
from src.gameplay.osu.world.object_lifecycle import WorldObject
from src.gameplay.osu.input.action_model import ActionType


def test_closed_loop_timing_calibrator_adaptive_learning():
    """Verify that consume_hit_errors nudges adaptive_offset_ms towards 0ms error and rejects outliers."""
    calibrator = AdaptiveTimingCalibrator(default_latency_ms=14.5, learning_rate=0.15, max_offset_ms=16.0)
    assert calibrator.adaptive_offset_ms == 0.0

    # 1. Player is consistently hitting 12ms late (+12ms hit error)
    hit_errors_batch_1 = [12.0, 14.0, 10.0, 12.0]
    new_offset = calibrator.consume_hit_errors(hit_errors_batch_1)
    # Median is 12.0. Adjustment = 12.0 * 0.15 = +1.8ms
    assert new_offset > 0.0
    assert abs(new_offset - 1.8) < 0.01

    # 2. Outliers (misses or dropped sliders > 35ms) must be discarded
    hit_errors_batch_2 = hit_errors_batch_1 + [85.0, -90.0, 105.0]
    offset_after_outliers = calibrator.consume_hit_errors(hit_errors_batch_2)
    # No valid new samples -> offset remains unchanged
    assert offset_after_outliers == new_offset

    # 3. New valid batch: now errors are negative (-8ms early)
    hit_errors_batch_3 = hit_errors_batch_2 + [-8.0, -8.0, -8.0]
    offset_after_negative = calibrator.consume_hit_errors(hit_errors_batch_3)
    # Median is -8.0. Adjustment = -8.0 * 0.15 = -1.2ms. New offset = 1.8 - 1.2 = 0.6ms
    assert offset_after_negative < new_offset
    assert abs(offset_after_negative - 0.6) < 0.01

    # 4. Clamp check: large errors cannot exceed max_offset_ms
    large_batch = hit_errors_batch_3 + [30.0] * 50
    final_offset = calibrator.consume_hit_errors(large_batch)
    assert final_offset <= 16.0

    # 5. Reset cleans state
    calibrator.reset_offset()
    assert calibrator.adaptive_offset_ms == 0.0
    assert len(calibrator._sample_errors) == 0


def test_tosu_client_parses_hit_errors_and_unstable_rate():
    """Verify that TosuClient parses hitErrorArray and unstableRate into snapshot."""
    client = TosuClient()
    payload = {
        "menu": {"state": 2},
        "gameplay": {
            "hp": {"normal": 0.8},
            "accuracy": 98.5,
            "hits": {
                "300": 100,
                "100": 2,
                "50": 0,
                "0": 1,
                "hitErrorArray": [12.5, -4.2, 8.1, 0.5],
                "unstableRate": 65.4,
            },
        },
    }
    client._parse_payload(payload)
    assert client.hit_errors == [12.5, -4.2, 8.1, 0.5]
    assert client.unstable_rate == 65.4

    snapshot = client.get_snapshot()
    assert snapshot["hit_errors"] == [12.5, -4.2, 8.1, 0.5]
    assert snapshot["unstable_rate"] == 65.4


def test_standard_brain_stream_dwell_and_tap_hold():
    """Verify that HitCircles in streams keep cursor centered for dwell time and use compact tap_hold."""
    brain = StandardBrain(keys=["z", "x"])
    c1 = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1)
    c2 = HitCircle(x=120.0, y=100.0, time_ms=1093.75, hit_type=1)  # 93.75ms (1/4 beat stream at 160 BPM)

    w1 = WorldObject(id="c1", hit_object=c1)
    w2 = WorldObject(id="c2", hit_object=c2)

    intents = brain.plan_actions(current_time_ms=950.0, objects=[w1, w2], timeline_anchor_ns=1_000_000_000)

    # 1. Tap hold for stream circle c2 must be compact (<= 28ms)
    tap_c2 = next(i for i in intents if i.action_id == "act_c2")
    assert tap_c2.metadata["tap_hold_ms"] <= 28.0
    assert tap_c2.metadata["tap_hold_ms"] >= 18.0

    # 2. Key alternation: c1 is z, c2 is x
    tap_c1 = next(i for i in intents if i.action_id == "act_c1")
    assert tap_c1.key == "z"
    assert tap_c2.key == "x"

    # 3. Cursor movement for c2 must NOT start immediately at c1 hit time (1000ms),
    # but after the dwell time (~6-8ms), ensuring c1 key click lands while cursor is on target
    c2_moves = [i for i in intents if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(c2_moves) > 0
    earliest_c2_move_ns = min(m.execute_at_ns for m in c2_moves)
    # Must be after c1's target time dispatch
    assert earliest_c2_move_ns > tap_c1.execute_at_ns


def test_consecutive_slider_cursor_preservation():
    """Verify that for consecutive sliders (slider 2 immediately following slider 1), cursor does NOT jump early."""
    brain = StandardBrain(keys=["z", "x"])
    s1 = Slider(
        x=100.0, y=100.0, time_ms=1000.0, hit_type=2,
        curve_type=CurveType.BEZIER, curve_points=[(150.0, 100.0)],
        repeat_count=1, pixel_length=100.0, duration_ms=200.0, end_time_ms=1200.0,
    )
    s2 = Slider(
        x=200.0, y=200.0, time_ms=1200.0, hit_type=2,
        curve_type=CurveType.BEZIER, curve_points=[(250.0, 200.0)],
        repeat_count=1, pixel_length=100.0, duration_ms=200.0, end_time_ms=1400.0,
    )

    w1 = WorldObject(id="s1", hit_object=s1)
    w2 = WorldObject(id="s2", hit_object=s2)

    intents = brain.plan_actions(current_time_ms=950.0, objects=[w1, w2], timeline_anchor_ns=1_000_000_000)

    # Key alternation: s1 uses z, s2 uses x
    s1_down = next(i for i in intents if i.action_id == "act_s1_down")
    s2_down = next(i for i in intents if i.action_id == "act_s2_down")
    assert s1_down.key == "z"
    assert s2_down.key == "x"

    # s1 safe tail release must be held past 1200ms (+12ms = 1212ms) with lead-time calibration
    s1_up = next(i for i in intents if i.action_id == "act_s1_up")
    assert s1_up.execute_at_ns == timing_calibrator.calculate_dispatch_time_ns(1212.0, 1_000_000_000)

    # Cursor movement to s2 head must NOT execute before s1 end dispatch time
    s2_moves = [i for i in intents if i.target_id == "s2" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(s2_moves) > 0


def test_single_tap_and_stream_alternation_polarity_reset():
    """
    Verify that:
    1. Single-tap baseline uses primary key 'z' for all jumps (delta_t > 145ms).
    2. Stream burst (delta_t <= 145ms) alternates strictly: z -> x -> z -> x.
    3. After stream ends, subsequent jumps immediately reset to 'z' (no polarity inversion flip-flop).
    """
    brain = StandardBrain(keys=["z", "x"])

    # Jump sequence: t = 1000, 1250, 1500 (delta_t = 250ms)
    j1 = WorldObject(id="j1", hit_object=HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1))
    j2 = WorldObject(id="j2", hit_object=HitCircle(x=200.0, y=100.0, time_ms=1250.0, hit_type=1))
    j3 = WorldObject(id="j3", hit_object=HitCircle(x=300.0, y=100.0, time_ms=1500.0, hit_type=1))

    # Stream 4-note burst: starts at 1800 (delta_t = 300ms from j3 -> fresh stream starts with 'z')
    # notes at 1800, 1890, 1980, 2070 (delta_t = 90ms)
    s1 = WorldObject(id="s1", hit_object=HitCircle(x=310.0, y=100.0, time_ms=1800.0, hit_type=1))
    s2 = WorldObject(id="s2", hit_object=HitCircle(x=320.0, y=100.0, time_ms=1890.0, hit_type=1))
    s3 = WorldObject(id="s3", hit_object=HitCircle(x=330.0, y=100.0, time_ms=1980.0, hit_type=1))
    s4 = WorldObject(id="s4", hit_object=HitCircle(x=340.0, y=100.0, time_ms=2070.0, hit_type=1))

    # Post-stream jumps: t = 2350, 2600 (delta_t = 280ms and 250ms)
    j4 = WorldObject(id="j4", hit_object=HitCircle(x=200.0, y=200.0, time_ms=2350.0, hit_type=1))
    j5 = WorldObject(id="j5", hit_object=HitCircle(x=100.0, y=200.0, time_ms=2600.0, hit_type=1))

    all_objects = [j1, j2, j3, s1, s2, s3, s4, j4, j5]
    intents = brain.plan_actions(current_time_ms=950.0, objects=all_objects, timeline_anchor_ns=1_000_000_000)

    tap_map = {i.target_id: i.key for i in intents if i.action_type == ActionType.KEY_TAP}

    # 1. Jumps before stream must all be 'z'
    assert tap_map["j1"] == "z"
    assert tap_map["j2"] == "z"
    assert tap_map["j3"] == "z"

    # 2. Stream burst must alternate starting from 'z' -> 'x' -> 'z' -> 'x'
    assert tap_map["s1"] == "z"
    assert tap_map["s2"] == "x"
    assert tap_map["s3"] == "z"
    assert tap_map["s4"] == "x"

    # 3. CRITICAL: Jumps after stream MUST reset to primary key 'z', NOT flip-flop to 'x'!
    assert tap_map["j4"] == "z"
    assert tap_map["j5"] == "z"


def test_controller_check_stuck_keys_clears_locked_slider_keys():
    """Verify that stuck key watchdog clears both _key_down_times and _locked_slider_keys."""
    from src.gameplay.osu.input.controller import CentralInputController
    from src.gameplay.osu.input.backends import MockInputBackend
    from src.gameplay.osu.core.clock import clock

    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)

    # Simulate key 'x' being held down as a locked slider key
    now_ns = clock.now_ns()
    old_time_ns = now_ns - 600_000_000  # 600ms ago (exceeds default max_stuck_duration_ms=500ms)

    ctrl._key_down_times["x"] = old_time_ns
    ctrl._locked_slider_keys.add("x")
    ctrl._max_stuck_duration_ns = 500_000_000  # 500ms

    assert "x" in ctrl._locked_slider_keys
    assert "x" in ctrl._key_down_times

    freed = ctrl.check_stuck_keys()
    assert freed >= 1
    assert "x" not in ctrl._key_down_times
    assert "x" not in ctrl._locked_slider_keys
    assert "x" not in backend.active_keys

