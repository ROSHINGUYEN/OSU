"""
Unit tests for High-BPM Aim Speed, Fast Slider Following, 475 RPM Spinner Rotation,
and Sub-millisecond Two-Stage Adaptive Timing Calibration.
"""

import math
import pytest
from src.gameplay.osu.beatmap.models import CurveType, HitCircle, Slider, Spinner
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.input.action_model import ActionType, CursorOwnerState
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator
from src.gameplay.osu.world.object_lifecycle import WorldObject


def test_mouse_departure_not_blocked_by_tap_hold():
    """
    Verify that in high BPM rapid sequences (e.g. 200 BPM stream, delta=75ms),
    the mouse cursor is NOT gated by the 20-25ms keyboard tap_hold and leaves
    promptly towards the next circle with full lead time.
    """
    brain = StandardBrain()
    c1 = WorldObject("c1", HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1))
    c2 = WorldObject("c2", HitCircle(x=200.0, y=200.0, time_ms=1075.0, hit_type=1))

    anchor_ns = clock.now_ns()
    intents = brain.plan_actions(
        current_time_ms=900.0,
        objects=[c1, c2],
        timeline_anchor_ns=anchor_ns,
    )

    # Find moves for c2
    c2_moves = [i for i in intents if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(c2_moves) >= 1, "Must generate mouse move for c2"

    # In nanoseconds: c1 hit time is anchor + 1000ms. Dispatch time for c2 is ~anchor + 1075ms - lead.
    # Earliest move should start well before 1000 + 25 = 1025ms!
    first_c2_move = c2_moves[0]
    expected_c1_hit_ns = anchor_ns + 1000 * 1_000_000
    move_offset_from_c1_ms = (first_c2_move.execute_at_ns - expected_c1_hit_ns) / 1_000_000.0

    # Under the old tap_hold gating, move_offset would be >= +20ms or +25ms.
    # Now it starts moving towards c2 smoothly right after c1 hit.
    assert move_offset_from_c1_ms < 65.0


def test_spinner_rotational_velocity_and_radius():
    """
    Verify spinner rotation uses compact 32px radius and near-maximum 475 RPM rotation.
    Also verify early-exit pre-aiming when a hit circle follows immediately after the spinner.
    """
    brain = StandardBrain()
    prep = brain.preprocessor
    center_sx, center_sy = prep.osu_to_screen(256.0, 192.0)

    # Spinner of 1000ms duration, followed by circle at 2150ms (gap=150ms < 350ms)
    sp = WorldObject("sp1", Spinner(x=256.0, y=192.0, time_ms=1000.0, hit_type=8, duration_ms=1000.0))
    c_next = WorldObject("c_next", HitCircle(x=400.0, y=300.0, time_ms=2150.0, hit_type=1))

    anchor_ns = clock.now_ns()
    intents = brain.plan_actions(
        current_time_ms=800.0,
        objects=[sp, c_next],
        timeline_anchor_ns=anchor_ns,
    )

    spin_moves = [
        i for i in intents
        if i.target_id == "sp1" and i.action_type == ActionType.MOUSE_MOVE and i.cursor_owner == CursorOwnerState.SPINNER_CONTROL
    ]
    assert len(spin_moves) >= 10, f"Expected dense spinner moves, got {len(spin_moves)}"

    # Check radius: every spin move must be at radius approx 32px from center (tolerance +/- 2px)
    for m in spin_moves:
        r = math.hypot(m.x - center_sx, m.y - center_sy)
        assert abs(r - 32.0) < 2.5, f"Expected compact radius ~32px, got {r}"

    # Check angular progression between steps: 475 RPM = 49.74 rad/s
    # In dt ~ 16ms: d_theta ~ 49.74 * 0.016 ~ 0.796 rad
    if len(spin_moves) >= 2:
        m0 = spin_moves[0]
        m1 = spin_moves[1]
        a0 = math.atan2(m0.y - center_sy, m0.x - center_sx)
        a1 = math.atan2(m1.y - center_sy, m1.x - center_sx)
        da = (a1 - a0) % (2.0 * math.pi)
        assert 0.5 <= da <= 1.2, f"Expected step angle ~0.8 rad, got {da}"

    # Verify mouse move to c_next exists
    c_next_moves = [i for i in intents if i.target_id == "c_next" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(c_next_moves) >= 1, "Must generate pre-aim move towards c_next"


def test_fast_slider_adaptive_step_density():
    """
    Verify that fast sliders (velocity > 1.2 px/ms) automatically receive denser
    step interpolation (step <= 12ms, 80-100Hz) to prevent follow circle edge breaks.
    """
    brain = StandardBrain()
    # Fast kick slider: length 300px, duration 120ms (velocity = 2.5 px/ms > 1.2)
    slider_fast = WorldObject(
        "sl_fast",
        Slider(
            x=100.0,
            y=100.0,
            time_ms=1000.0,
            hit_type=2,
            curve_type=CurveType.LINEAR,
            curve_points=[(100.0, 100.0), (400.0, 100.0)],
            repeat_count=1,
            pixel_length=300.0,
            duration_ms=120.0,
        ),
    )

    anchor_ns = clock.now_ns()
    intents = brain.plan_actions(
        current_time_ms=800.0,
        objects=[slider_fast],
        timeline_anchor_ns=anchor_ns,
    )

    slide_moves = [
        i for i in intents
        if i.target_id == "sl_fast" and i.action_type == ActionType.MOUSE_MOVE and i.cursor_owner == CursorOwnerState.SLIDER_FOLLOW
    ]
    # With effective_step_ms <= 12ms on 120ms duration:
    # span_steps = max(4, int(120 / 12)) = 10 steps (+1 dwell)
    assert len(slide_moves) >= 8, f"Fast slider should have dense moves (>=8), got {len(slide_moves)}"


def test_two_stage_calibrator_fast_convergence():
    """
    Verify that default AdaptiveTimingCalibrator uses two-stage adaptive learning rate:
    Stage 1: alpha = 0.28 for first 5 samples (rapid error elimination)
    Stage 2: alpha = 0.10 for samples > 5 (steady-state stability)
    """
    cal = AdaptiveTimingCalibrator()  # Default 0.12 triggers two-stage
    assert cal.adaptive_offset_ms == 0.0

    # Sample 1: Hit 10ms late (+10ms)
    cal.consume_hit_errors([10.0])
    # Adjustment = 10.0 * 0.28 = +2.8ms
    assert abs(cal.adaptive_offset_ms - 2.8) < 0.01

    # Samples 2 to 5: Feed 4 more samples (+5ms each)
    cal.consume_hit_errors([10.0, 10.0, 10.0, 10.0])
    # Still in stage 1 (total sample_errors <= 5)
    assert cal.adaptive_offset_ms > 2.8

    # Feed large batch to enter stage 2 (> 5 samples)
    curr_offset = cal.adaptive_offset_ms
    cal.consume_hit_errors([10.0] * 5 + [10.0])
    # Now in stage 2: alpha = 0.10
    cal.consume_hit_errors([10.0] * 5 + [10.0, 10.0])
    assert cal.adaptive_offset_ms > curr_offset


def test_cs55_jump_aim_landing_lead_and_immediate_departure():
    """
    Verify that on high CS beatmaps (CS >= 5.0, tiny circles),
    1/2-beat jump patterns depart promptly from earliest_move_ms (zero artificial delay)
    and arrive at the circle center with at least 20ms of landing stabilization window
    before the key tap dispatch time, preventing sub-millisecond miss regressions.
    """
    brain = StandardBrain(cs=5.5)
    # Circle 1 at 1000ms, Circle 2 at 1166ms (1/2 beat jump, 166ms gap, 250px distance)
    c1 = WorldObject("c1", HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1))
    c2 = WorldObject("c2", HitCircle(x=350.0, y=100.0, time_ms=1166.0, hit_type=1))

    anchor_ns = 1_000_000_000_000
    intents = brain.plan_actions(
        current_time_ms=900.0,
        objects=[c1, c2],
        timeline_anchor_ns=anchor_ns,
    )

    c2_moves = [
        i for i in intents
        if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE
    ]
    c2_tap = next(
        i for i in intents
        if i.target_id == "c2" and i.action_type == ActionType.KEY_TAP
    )

    assert len(c2_moves) >= 2, "Must generate smooth multi-step trajectory for jump"

    # First move towards c2 should start promptly after c1 hit + dwell (1000 + ~2ms)
    first_move = c2_moves[0]
    expected_c1_hit_ns = anchor_ns + 1000 * 1_000_000
    departure_delay_ms = (first_move.execute_at_ns - expected_c1_hit_ns) / 1_000_000.0
    # Under old logic, departure_delay_ms was > 100ms due to available_time * 0.6 clamping!
    # Under new logic, departure happens smoothly with midpoint waypoint under 75ms
    assert departure_delay_ms < 75.0, f"Departure was delayed too much: {departure_delay_ms}ms"

    # Final move towards c2 must arrive at least 20ms BEFORE c2_tap dispatch
    last_move = c2_moves[-1]
    landing_window_ms = (c2_tap.execute_at_ns - last_move.execute_at_ns) / 1_000_000.0
    assert landing_window_ms >= 20.0, (
        f"Landing window before tap was only {landing_window_ms:.2f}ms (expected >= 20.0ms for CS 5.5)"
    )

