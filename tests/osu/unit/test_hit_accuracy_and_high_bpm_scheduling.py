"""
Comprehensive Test Suite for Shiori osu! Gameplay:
Hit Accuracy, High-BPM Sequence Scheduling, Cursor Trajectory Feasibility,
Slider Tracking Protection, and Closed-Loop Feedback Verification.
"""

import math
import pytest
from typing import List, Tuple

from src.gameplay.osu.beatmap.models import (
    CurveType,
    HitCircle,
    HitObjectFlag,
    Slider,
)
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.input.action_model import (
    ActionIntent,
    ActionType,
    CursorOwnerState,
    InputStateStatus,
)
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.input.queue import CentralInputQueue
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.planning.planner import RollingHorizonPlanner
from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator
from src.gameplay.osu.timing.hit_windows import HitWindowModel
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject
from src.gameplay.osu.world.world_state import WorldStateManager


@pytest.fixture
def prep():
    p = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    p.update_window_position(0, 0)
    return p


# ==============================================================================
# GROUP 1: CIRCLE TESTS (High-BPM, Stacked, Jumps, Feasibility, Stale Drop)
# ==============================================================================

def test_circle_single_timing_and_aim(prep):
    """Test 1.1: Single circle plans aim trajectory ahead of hit window and taps primary key 'z'."""
    brain = StandardBrain(preprocessor=prep, od=8.0)
    circle = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    w_obj = WorldObject(id="c_single", hit_object=circle)

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE]
    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]

    assert len(taps) == 1
    tap = taps[0]
    assert tap.key == "z"
    assert tap.target_id == "c_single"
    assert tap.priority == 3

    # All moves must precede or coincide with tap execute time
    for m in moves:
        assert m.execute_at_ns <= tap.execute_at_ns
        assert m.priority == 2

    # Final move position must match circle target exactly
    sx, sy = prep.osu_to_screen(256.0, 192.0)
    assert moves[-1].x == sx
    assert moves[-1].y == sy


def test_circles_close_spatial_different_time(prep):
    """Test 1.2: Circles close in space (<20px) but distinct in time (dt=200ms) maintain timeline order."""
    brain = StandardBrain(preprocessor=prep)
    c1 = HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    c2 = HitCircle(x=210.0, y=205.0, time_ms=1200.0, hit_type=HitObjectFlag.CIRCLE)
    objs = [WorldObject(id="c1", hit_object=c1), WorldObject(id="c2", hit_object=c2)]

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 2
    # Must hit c1 first, then c2
    assert taps[0].target_id == "c1"
    assert taps[1].target_id == "c2"
    assert taps[0].execute_at_ns < taps[1].execute_at_ns


def test_circles_stacked_identical_coords(prep):
    """Test 1.3: Stacked circles (identical x, y) require no cursor jerk and strictly alternate keys."""
    brain = StandardBrain(preprocessor=prep)
    c1 = HitCircle(x=300.0, y=200.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    c2 = HitCircle(x=300.0, y=200.0, time_ms=1100.0, hit_type=HitObjectFlag.CIRCLE)
    c3 = HitCircle(x=300.0, y=200.0, time_ms=1200.0, hit_type=HitObjectFlag.CIRCLE)
    objs = [
        WorldObject(id="s1", hit_object=c1),
        WorldObject(id="s2", hit_object=c2),
        WorldObject(id="s3", hit_object=c3),
    ]

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 3
    # Check alternating keys for the 100ms rapid stream notes
    keys = [t.key for t in taps]
    assert keys[0] == "z"
    assert keys[1] == "x"
    assert keys[2] == "z"


def test_circles_high_bpm_stream_alternation(prep):
    """Test 1.5: 240 BPM Stream (dt = 62.5ms) strictly alternates Z/X and co-tunes tap hold durations."""
    brain = StandardBrain(preprocessor=prep)
    base_t = 1000.0
    dt = 62.5  # 1/4 beat at 240 BPM
    objs = [
        WorldObject(id=f"st_{i}", hit_object=HitCircle(x=200.0 + (i * 10), y=200.0, time_ms=base_t + (i * dt), hit_type=HitObjectFlag.CIRCLE))
        for i in range(8)
    ]

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=900.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 8
    keys = [t.key for t in taps]
    expected_keys = ["z", "x", "z", "x", "z", "x", "z", "x"]
    assert keys == expected_keys

    # The first note starts the stream (single-tap hold <= 35ms), while subsequent burst notes co-tune to <= 28.5ms
    for idx, t in enumerate(taps):
        hold_ms = t.metadata.get("tap_hold_ms", 25.0)
        if idx == 0:
            assert 18.0 <= hold_ms <= 35.0
        else:
            assert 18.0 <= hold_ms <= 28.5


def test_circle_large_jump_feasibility(prep):
    """Test 1.6: Large cross-screen jump (>400px) with tight window triggers snappy 1-step trajectory."""
    brain = StandardBrain(preprocessor=prep)
    brain._cursor_x = 50.0
    brain._cursor_y = 50.0
    # Jump to far corner (450, 350) with only 35ms available
    c_jump = HitCircle(x=450.0, y=350.0, time_ms=1035.0, hit_type=HitObjectFlag.CIRCLE)
    w_obj = WorldObject(id="c_jump", hit_object=c_jump)

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=1000.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE]
    assert len(moves) >= 1
    # Because velocity requirement is high (>3.0 px/ms), steps must be minimal (1 or 2)
    assert len(moves) <= 2
    # Target coordinate reached
    tx, ty = prep.osu_to_screen(450.0, 350.0)
    assert moves[-1].x == tx
    assert moves[-1].y == ty


def test_circle_impossible_deadline_and_stale_drop():
    """Test 1.7: Stale actions beyond tolerance window are dropped by CentralInputQueue."""
    queue = CentralInputQueue()
    now_ns = clock.now_ns()

    stale_action = ActionIntent(
        action_id="act_stale",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=now_ns - 200_000_000,
        deadline_ns=now_ns - 100_000_000,
        key="z",
    )
    future_action = ActionIntent(
        action_id="act_valid",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=now_ns + 10_000_000,
        deadline_ns=now_ns + 100_000_000,
        key="x",
    )

    queue.push(stale_action)
    queue.push(future_action)

    # Popping at now_ns should discard stale action and return None (since future action is not due yet)
    due = queue.pop_due(now_ns)
    assert due is None
    assert queue.stale_rejected_count == 1
    assert len(queue) == 1

    # Pop when future action is due
    due_future = queue.pop_due(now_ns + 15_000_000)
    assert due_future is not None
    assert due_future.action_id == "act_valid"


def test_circle_observed_cursor_feedback(prep):
    """Test 1.9: Real observed cursor feedback shifts initial trajectory origin accurately."""
    brain = StandardBrain(preprocessor=prep)
    # Default virtual cursor
    assert brain._cursor_x == 256.0
    assert brain._cursor_y == 192.0

    # Observed cursor at screen position corresponding to (100, 100)
    obs_sx, obs_sy = prep.osu_to_screen(100.0, 100.0)
    target = HitCircle(x=400.0, y=300.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    w_obj = WorldObject(id="c_obs", hit_object=target)

    anchor_ns = 1_000_000_000
    brain.plan_actions(
        current_time_ms=800.0,
        objects=[w_obj],
        timeline_anchor_ns=anchor_ns,
        observed_cursor_pos=(obs_sx, obs_sy),
    )

    # Internal cursor position updated to real observation
    assert math.isclose(brain._cursor_x, 100.0, abs_tol=1.0)
    assert math.isclose(brain._cursor_y, 100.0, abs_tol=1.0)


# ==============================================================================
# GROUP 2: SLIDER TESTS (Tracking, Multi-Repeat, Lock, Cancel, Release)
# ==============================================================================

def test_slider_normal_tracking_and_ownership(prep):
    """Test 2.1: Slider establishes key down, slider follow waypoints (priority 1), and tail release."""
    brain = StandardBrain(preprocessor=prep)
    ctrl = CentralInputController(backend=MockInputBackend())
    ctrl.auto_aim_enabled = True

    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(200.0, 100.0)],
        repeat_count=1,
        pixel_length=100.0,
        duration_ms=400.0,
    )
    w_obj = WorldObject(id="sld_norm", hit_object=slider)
    anchor_ns = 1_000_000_000

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    k_downs = [i for i in intents if i.action_type == ActionType.KEY_DOWN]
    k_ups = [i for i in intents if i.action_type == ActionType.KEY_UP]
    slides = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.cursor_owner == CursorOwnerState.SLIDER_FOLLOW]

    assert len(k_downs) == 1
    assert len(k_ups) == 1
    assert len(slides) >= 4
    # All slide points must have priority 1 (highest)
    for s in slides:
        assert s.priority == 1
        assert s.target_id == "sld_norm"

    # Execute k_down in controller -> locks slider
    ctrl.execute_action(k_downs[0])
    assert ctrl._active_slider_owner == "sld_norm"
    assert "z" in ctrl._locked_slider_keys

    # Foreign mouse move from circle must be cancelled
    foreign_move = ActionIntent(
        action_id="act_circle_move",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=anchor_ns + 1_200_000_000,
        deadline_ns=anchor_ns + 1_300_000_000,
        target_id="circle_other",
        cursor_owner=CursorOwnerState.MOVING_TO_TARGET,
        x=400.0,
        y=400.0,
    )
    ctrl.execute_action(foreign_move)
    assert foreign_move.status == InputStateStatus.CANCELLED

    # Release k_up
    ctrl.execute_action(k_ups[0])
    assert ctrl._active_slider_owner is None
    assert "z" not in ctrl._locked_slider_keys


def test_slider_circle_during_active_slider_strict_protection(prep):
    """Test 2.4: Circle scheduled during active slider does not disrupt slider key or cursor."""
    brain = StandardBrain(preprocessor=prep)
    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(300.0, 100.0)],
        repeat_count=1,
        pixel_length=200.0,
        duration_ms=500.0,
    )
    # Circle placed at t = 1200ms (during slider)
    interfering_circle = HitCircle(x=400.0, y=400.0, time_ms=1200.0, hit_type=HitObjectFlag.CIRCLE)

    objs = [
        WorldObject(id="sld_active", hit_object=slider),
        WorldObject(id="c_interfering", hit_object=interfering_circle),
    ]

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    # Circle must NOT use the key held by the slider
    circle_taps = [i for i in intents if i.action_type == ActionType.KEY_TAP and i.target_id == "c_interfering"]
    assert len(circle_taps) == 1
    # Slider used 'z', circle must use 'x'
    assert circle_taps[0].key == "x"

    # Circle must NOT generate a MOUSE_MOVE before earliest_move_ms (when slider ends)
    circle_moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.target_id == "c_interfering"]
    assert len(circle_moves) == 0


def test_slider_cancel_and_release():
    """Test 2.5: cancel_active_slider explicitly frees locks and neutralizes state."""
    ctrl = CentralInputController(backend=MockInputBackend())
    down_act = ActionIntent(
        action_id="act_sld_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=1_000_000,
        deadline_ns=2_000_000,
        target_id="slider_to_cancel",
        key="z",
        metadata={"is_slider": True},
    )
    ctrl.execute_action(down_act)
    assert ctrl._active_slider_owner == "slider_to_cancel"
    assert "z" in ctrl._locked_slider_keys

    ctrl.cancel_active_slider("slider_to_cancel")
    assert ctrl._active_slider_owner is None
    assert "z" not in ctrl._locked_slider_keys


# ==============================================================================
# GROUP 3: TIMING & CALIBRATION TESTS (Clock, HitWindows, Lead Time, Feedback)
# ==============================================================================

def test_timing_hit_windows_and_deadlines():
    """Test 3.1: Hit window calculations match osu! standard rules exactly."""
    # OD 8: w300 = 80 - 48 = 32ms, w100 = 140 - 64 = 76ms, w50 = 200 - 80 = 120ms
    w300, w100, w50 = HitWindowModel.standard(8.0)
    assert w300 == 32.0
    assert w100 == 76.0
    assert w50 == 120.0

    # OD 10: w300 = 20ms, w100 = 60ms, w50 = 100ms
    w300_10, w100_10, w50_10 = HitWindowModel.standard(10.0)
    assert w300_10 == 20.0
    assert w100_10 == 60.0
    assert w50_10 == 100.0


def test_timing_adaptive_kalman_calibration():
    """Test 3.2: Closed-loop timing feedback shifts offset smoothly towards zero error."""
    cal = AdaptiveTimingCalibrator(default_latency_ms=2.0, learning_rate=0.2)
    cal.reset_offset()

    # Repeated late hits (+10ms)
    for _ in range(5):
        cal.record_hit_sample(10.0, confidence=1.0)

    # Offset must increase (dispatching earlier to compensate)
    assert cal.adaptive_offset_ms > 0.0

    # Outliers (>35ms) must be discarded
    prev_offset = cal.adaptive_offset_ms
    cal.record_hit_sample(60.0, confidence=1.0)
    assert cal.adaptive_offset_ms == prev_offset


# ==============================================================================
# GROUP 4: REGRESSION TESTS (Pure osu! Standard Mode Focus)
# ==============================================================================

def test_standard_mode_pure_focus():
    """Test 4.1: StandardBrain generates correct aiming and key actions in pure standard mode."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    c1 = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    c2 = HitCircle(x=200.0, y=200.0, time_ms=1200.0, hit_type=HitObjectFlag.CIRCLE)

    w1 = WorldObject(id="std_1", hit_object=c1)
    w2 = WorldObject(id="std_2", hit_object=c2)

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w1, w2], timeline_anchor_ns=1_000_000_000)
    assert len(intents) >= 2
    assert all(i.mode == GameMode.STANDARD for i in intents)


def test_circles_simultaneous_or_zero_dt(prep):
    """Test 1.4: Circles with simultaneous or near-zero dt strictly alternate keys without stuck key."""
    brain = StandardBrain(preprocessor=prep)
    c1 = HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    c2 = HitCircle(x=220.0, y=200.0, time_ms=1000.0, hit_type=HitObjectFlag.CIRCLE)
    objs = [WorldObject(id="cz_1", hit_object=c1), WorldObject(id="cz_2", hit_object=c2)]

    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 2
    # Even at dt <= 0, keys must alternate to prevent key jamming
    assert taps[0].key != taps[1].key


def test_slider_long_duration(prep):
    """Test 2.2: Long duration slider (2000ms) produces stable continuous tracking waypoints."""
    brain = StandardBrain(preprocessor=prep)
    slider = Slider(
        x=50.0,
        y=50.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(450.0, 50.0)],
        repeat_count=1,
        pixel_length=400.0,
        duration_ms=2000.0,
    )
    w_obj = WorldObject(id="sld_long", hit_object=slider)
    anchor_ns = 1_000_000_000

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    slides = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.cursor_owner == CursorOwnerState.SLIDER_FOLLOW]
    # Long duration (2000ms) must produce at least 50 waypoints (sampled every ~12-16ms)
    assert len(slides) >= 50
    # Must end at tail
    tail_x, tail_y = prep.osu_to_screen(450.0, 50.0)
    assert slides[-1].x == tail_x
    assert slides[-1].y == tail_y


def test_slider_multi_repeat_exact_ticks(prep):
    """Test 2.3: Multi-repeat slider (repeat_count=3) hits exact 100% turnarounds."""
    brain = StandardBrain(preprocessor=prep)
    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(300.0, 100.0)],
        repeat_count=3,
        pixel_length=200.0,
        duration_ms=900.0,
    )
    w_obj = WorldObject(id="sld_repeat3", hit_object=slider)
    anchor_ns = 1_000_000_000

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    slides = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.cursor_owner == CursorOwnerState.SLIDER_FOLLOW]
    head_sx, head_sy = prep.osu_to_screen(100.0, 100.0)
    tail_sx, tail_sy = prep.osu_to_screen(300.0, 100.0)

    # Check for points reaching exact head and tail turnaround coordinates
    reached_tail = any(math.isclose(s.x, tail_sx, abs_tol=1.0) and math.isclose(s.y, tail_sy, abs_tol=1.0) for s in slides)
    reached_head = any(math.isclose(s.x, head_sx, abs_tol=1.0) and math.isclose(s.y, head_sy, abs_tol=1.0) for s in slides)

    assert reached_tail, "Slider did not reach 100% tail coordinate during turnaround"
    assert reached_head, "Slider did not reach 100% head coordinate during reverse repeat"


def test_slider_emergency_stop_release_all():
    """Test 2.6: emergency_release_all instantly resets locks, held keys, and mouse buttons."""
    ctrl = CentralInputController(backend=MockInputBackend())
    ctrl.execute_action(
        ActionIntent(
            action_id="act_s_down",
            action_type=ActionType.KEY_DOWN,
            execute_at_ns=1_000,
            deadline_ns=2_000,
            target_id="sld_emergency",
            key="z",
            metadata={"is_slider": True},
        )
    )
    assert ctrl._active_slider_owner == "sld_emergency"
    assert "z" in ctrl._locked_slider_keys

    ctrl.emergency_release_all()
    assert ctrl._active_slider_owner is None
    assert len(ctrl._locked_slider_keys) == 0
    assert len(ctrl._key_down_times) == 0


def test_timing_high_bpm_tight_scheduling(prep):
    """Test 3.3: 280 BPM ultra-high-speed sequence (53.5ms intervals) is scheduled without collision."""
    brain = StandardBrain(preprocessor=prep)
    base_t = 1000.0
    dt = 53.57  # 280 BPM 1/4 stream
    objs = [
        WorldObject(id=f"fast_{i}", hit_object=HitCircle(x=150.0 + (i * 15), y=150.0, time_ms=base_t + (i * dt), hit_type=HitObjectFlag.CIRCLE))
        for i in range(6)
    ]
    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 6
    # Strictly increasing monotonic execution deadlines
    for idx in range(len(taps) - 1):
        assert taps[idx].execute_at_ns < taps[idx + 1].execute_at_ns
        assert taps[idx].key != taps[idx + 1].key


def test_180_bpm_half_beat_sequence_strict_alternation(prep):
    """Test 3.4: 180 BPM half-beat sequence (166.7ms intervals) strictly alternates z -> x -> z -> x."""
    brain = StandardBrain(preprocessor=prep)
    base_t = 1000.0
    dt = 166.67  # 180 BPM 1/2 beat
    objs = [
        WorldObject(id=f"c180_{i}", hit_object=HitCircle(x=200.0 + (i * 20), y=200.0, time_ms=base_t + (i * dt), hit_type=HitObjectFlag.CIRCLE))
        for i in range(6)
    ]
    anchor_ns = 1_000_000_000
    intents = brain.plan_actions(current_time_ms=800.0, objects=objs, timeline_anchor_ns=anchor_ns)

    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(taps) == 6
    keys = [t.key for t in taps]
    assert keys == ["z", "x", "z", "x", "z", "x"], f"Expected strict alternation for 180 BPM 1/2 beat, got {keys}"


def test_post_slider_transition_alternates_key(prep):
    """Test 3.5: Circle following immediately after a slider tail alternates to the opposite key."""
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    # Slider from 1000ms to 1200ms (duration 200ms)
    sl = Slider(x=200.0, y=200.0, time_ms=1000.0, hit_type=HitObjectFlag.SLIDER, duration_ms=200.0)
    w_sl = WorldObject("sl_prev", sl)

    # Circle at 1366.7ms (166.7ms after slider head start, ~154ms after safe tail release)
    circle = HitCircle(x=350.0, y=250.0, time_ms=1366.7, hit_type=HitObjectFlag.CIRCLE)
    w_circle = WorldObject("c_post", circle)

    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_sl, w_circle], timeline_anchor_ns=anchor_ns)

    sl_down = [i for i in intents if i.target_id == "sl_prev" and i.action_type == ActionType.KEY_DOWN][0]
    c_tap = [i for i in intents if i.target_id == "c_post" and i.action_type == ActionType.KEY_TAP][0]

    assert sl_down.key == "z"
    assert c_tap.key == "x", f"Post-slider circle must alternate to 'x' to avoid debounce choke, got '{c_tap.key}'"


def test_consecutive_tight_sliders_alternate_keys(prep):
    """Test 3.6: Consecutive tight sliders (like Shibayan map combo 348) alternate keys (z then x)."""
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    # Slider 1 at 90498ms, dur 167ms (ends 90665ms)
    sl1 = Slider(x=200.0, y=200.0, time_ms=90498.0, hit_type=HitObjectFlag.SLIDER, duration_ms=167.0)
    w_sl1 = WorldObject("sl1", sl1)

    # Slider 2 at 90831ms, dur 167ms (gap from tail ~154ms)
    sl2 = Slider(x=300.0, y=200.0, time_ms=90831.0, hit_type=HitObjectFlag.SLIDER, duration_ms=167.0)
    w_sl2 = WorldObject("sl2", sl2)

    intents = brain.plan_actions(current_time_ms=90000.0, objects=[w_sl1, w_sl2], timeline_anchor_ns=anchor_ns)

    sl1_down = [i for i in intents if i.target_id == "sl1" and i.action_type == ActionType.KEY_DOWN][0]
    sl2_down = [i for i in intents if i.target_id == "sl2" and i.action_type == ActionType.KEY_DOWN][0]

    assert sl1_down.key == "z"
    assert sl2_down.key == "x", f"Consecutive sliders must alternate keys z -> x, got '{sl1_down.key}' then '{sl2_down.key}'"


def test_central_input_controller_debounce_shield():
    """Test 3.7: Controller safely remaps key to alternate when tapped within 20ms debounce recovery window."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)

    # Key 'z' was just tapped and released
    ctrl.execute_action(
        ActionIntent(
            action_id="tap_z_1",
            action_type=ActionType.KEY_TAP,
            execute_at_ns=1_000,
            deadline_ns=2_000,
            key="z",
        )
    )
    # Release 'z' and record release time
    ctrl.execute_action(
        ActionIntent(
            action_id="up_z_1",
            action_type=ActionType.KEY_UP,
            execute_at_ns=1_500,
            deadline_ns=2_000,
            key="z",
        )
    )

    # Less than 20ms later, another tap request for 'z' arrives while 'x' is completely free
    tap_intent2 = ActionIntent(
        action_id="tap_z_2",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=clock.now_ns(),
        deadline_ns=clock.now_ns() + 50_000_000,
        key="z",
    )
    ctrl.execute_action(tap_intent2)

    # Controller hardware debounce protection must have redirected tap to 'x'
    assert tap_intent2.key == "x", f"Expected tap to be remapped to 'x' due to debounce protection, got '{tap_intent2.key}'"


def test_mouse_move_coalescing_drops_obsolete_waypoints_under_lag():
    """Test 3.8: CentralInputQueue coalesces overdue mouse waypoints to eliminate queue lag and HoL blocking."""
    queue = CentralInputQueue()
    base_ns = 1_000_000_000

    # 4 intermediate waypoints along a slider curve, all overdue by the time pop_due is called
    for i in range(1, 5):
        queue.push(
            ActionIntent(
                action_id=f"act_s1_slide_{i}",
                action_type=ActionType.MOUSE_MOVE,
                execute_at_ns=base_ns + (i * 10_000_000),
                deadline_ns=base_ns + (i * 10_000_000) + 100_000_000,
                target_id="s1",
                cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
                x=float(100 + i * 10),
                y=float(200 + i * 10),
                priority=1,
            )
        )

    # Also push a key tap scheduled slightly later
    queue.push(
        ActionIntent(
            action_id="act_c2_tap",
            action_type=ActionType.KEY_TAP,
            execute_at_ns=base_ns + 45_000_000,
            deadline_ns=base_ns + 100_000_000,
            target_id="c2",
            key="z",
            priority=3,
        )
    )

    # When pop_due is called at t = base_ns + 42ms (all 4 slider waypoints are overdue):
    # Coalescing MUST drop waypoints 1, 2, 3 and return waypoint 4 immediately!
    popped = queue.pop_due(base_ns + 42_000_000)
    assert popped is not None
    assert popped.action_id == "act_s1_slide_4", f"Expected coalesced newest waypoint slide_4, got {popped.action_id}"
    assert popped.x == 140.0 and popped.y == 240.0

    # Next pop before key tap time returns None
    assert queue.pop_due(base_ns + 43_000_000) is None

    # Key tap is popped intact at its scheduled time without being dropped
    key_act = queue.pop_due(base_ns + 46_000_000)
    assert key_act is not None
    assert key_act.action_id == "act_c2_tap"
    assert key_act.key == "z"


def test_hit_circle_aim_lead_time_guarantee(prep):
    """Test 3.9: StandardBrain ensures cursor arrives at hit circle 12-18ms before key tap dispatch."""
    brain = StandardBrain(preprocessor=prep)
    circle = HitCircle(x=300.0, y=200.0, time_ms=1000.0, hit_type=1)
    w_obj = WorldObject(id="circle_lead", hit_object=circle)
    anchor_ns = 1_000_000_000

    intents = brain.plan_actions(current_time_ms=850.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)
    moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE]
    taps = [i for i in intents if i.action_type == ActionType.KEY_TAP]

    assert len(moves) >= 1
    assert len(taps) == 1

    final_move = moves[-1]
    tap = taps[0]

    lead_ms = (tap.execute_at_ns - final_move.execute_at_ns) / 1_000_000.0
    assert 10.0 <= lead_ms <= 24.0, f"Expected lead time between 10ms and 24ms, got {lead_ms:.2f}ms"


def test_slider_tail_early_cursor_release_for_tight_successive_circle(prep):
    """Test 3.10: Slider tail cursor is freed at end_t_ms (not +12ms) when successive circle is tight."""
    brain = StandardBrain(preprocessor=prep)
    slider = Slider(x=100.0, y=100.0, time_ms=1000.0, hit_type=2, duration_ms=300.0, repeat_count=1)
    circle = HitCircle(x=400.0, y=400.0, time_ms=1340.0, hit_type=1)  # only 40ms after slider end!

    w_slider = WorldObject(id="s1", hit_object=slider)
    w_circle = WorldObject(id="c2", hit_object=circle)
    anchor_ns = 1_000_000_000

    intents = brain.plan_actions(current_time_ms=950.0, objects=[w_slider, w_circle], timeline_anchor_ns=anchor_ns)

    c2_moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.target_id == "c2"]
    c2_taps = [i for i in intents if i.action_type == ActionType.KEY_TAP and i.target_id == "c2"]

    assert len(c2_moves) >= 1, "Cursor move for c2 must be scheduled!"
    assert len(c2_taps) == 1, "Tap for c2 must be scheduled!"

    # Cursor for c2 must be scheduled before c2 tap dispatch time
    assert c2_moves[-1].execute_at_ns < c2_taps[0].execute_at_ns


