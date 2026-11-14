"""
Unit and Integration Tests for Shiori osu! Standard Slider Execution Pipeline.
==============================================================================
Validates:
1. Slider path and coordinate calculation (head, mid, repeat point, reverse, tail).
2. Circumscribed circle arc (P) orientation across all quadrants and reflex angles.
3. Multi-repeat turnaround points sampled at exact span boundaries.
4. Strict cursor ownership: foreign Hit Circles cannot hijack cursor during active slider.
5. CentralInputController cursor lock and foreign move suppression.
6. Stale action handling and queue ordering without premature jumping.
7. Playfield coordinate transformations with desktop window offsets.
8. Key-up synchronization with safe tail margin.
9. Structured slider diagnostics telemetry emission.
"""

import math
import time
from unittest.mock import MagicMock

import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.beatmap.models import CurveType, HitCircle, Slider
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    state_store,
)
from src.gameplay.osu.input.action_model import (
    ActionIntent,
    ActionType,
    CursorOwnerState,
    InputStateStatus,
)
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.input.queue import CentralInputQueue
from src.gameplay.osu.math.slider_curves import (
    calculate_slider_path,
    clip_points_to_length,
    sample_perfect_circle,
)
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


# ---------------------------------------------------------------------------
# 1. Path Geometry & Reflex Arc Invariance
# ---------------------------------------------------------------------------

def test_slider_head_mid_and_tail_path_accuracy():
    """Verify exact coordinates at head, midpoint, and tail for linear & bezier sliders."""
    # Linear slider from (100, 100) to (300, 100), length 200
    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=2,
        curve_type=CurveType.LINEAR,
        curve_points=[(300.0, 100.0)],
        repeat_count=1,
        pixel_length=200.0,
        duration_ms=400.0,
        end_time_ms=1400.0,
    )
    pts = calculate_slider_path(slider)
    assert len(pts) >= 2
    assert pts[0] == (100.0, 100.0)
    assert pts[-1] == (300.0, 100.0)

    brain = StandardBrain()
    head_x, head_y = brain._interpolate_slider_path(pts, 0.0, repeat_count=1)
    mid_x, mid_y = brain._interpolate_slider_path(pts, 0.5, repeat_count=1)
    tail_x, tail_y = brain._interpolate_slider_path(pts, 1.0, repeat_count=1)

    assert abs(head_x - 100.0) < 0.1 and abs(head_y - 100.0) < 0.1
    assert abs(mid_x - 200.0) < 0.1 and abs(mid_y - 100.0) < 0.1
    assert abs(tail_x - 300.0) < 0.1 and abs(tail_y - 100.0) < 0.1


def test_perfect_circle_reflex_arc_and_quadrant_invariance():
    """
    Verify sample_perfect_circle passes through intermediate control point P1
    for acute, obtuse, and reflex arcs (>180 deg) in all 4 quadrants.
    This directly regression-tests Root Cause 1.
    """
    # Test 1: Reflex arc (> 180 degrees) where old implementation took the 45-degree shortcut
    p0 = (100.0, 0.0)      # 0 deg
    p1 = (0.0, -100.0)     # 270 deg (-90 deg)
    p2 = (70.71, -70.71)   # 315 deg (-45 deg)
    pts_reflex = sample_perfect_circle(p0, p1, p2, pixel_length=0.0, num_points=35)

    # Arc MUST visit p1 (min distance to p1 must be < 10 px with 35 points)
    min_dist_to_p1 = min(math.hypot(x - p1[0], y - p1[1]) for x, y in pts_reflex)
    assert min_dist_to_p1 < 10.0, f"Arc bypassed P1! Min dist={min_dist_to_p1:.2f}px"

    # Test 2: Invariance across 8 cardinal semicircle orientations (both CW and CCW)
    for ang_start in [0, 90, 180, 270]:
        for direction in [1, -1]:
            a0 = math.radians(ang_start)
            a1 = a0 + direction * math.radians(90)
            a2 = a0 + direction * math.radians(180)
            p0 = (200.0 + 100.0 * math.cos(a0), 200.0 + 100.0 * math.sin(a0))
            p1 = (200.0 + 100.0 * math.cos(a1), 200.0 + 100.0 * math.sin(a1))
            p2 = (200.0 + 100.0 * math.cos(a2), 200.0 + 100.0 * math.sin(a2))
            pts = sample_perfect_circle(p0, p1, p2, pixel_length=100.0 * math.pi, num_points=25)
            mid = pts[len(pts) // 2]
            dist = math.hypot(mid[0] - p1[0], mid[1] - p1[1])
            assert dist < 1.0, f"Failed at start={ang_start} deg, dir={direction}: dist={dist:.2f}px"


# ---------------------------------------------------------------------------
# 2. Multi-Repeat Turnaround Points & Reversal
# ---------------------------------------------------------------------------

def test_multi_repeat_slider_hits_exact_turnarounds():
    """
    Verify that multi-repeat sliders (repeat_count=2, 3, 5) explicitly sample
    every repeat arrow turnaround point and the tail.
    """
    brain = StandardBrain()
    # 5-repeat slider (like in Camellia - Red Room at 184348ms)
    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=2,
        curve_type=CurveType.LINEAR,
        curve_points=[(200.0, 100.0)],
        repeat_count=5,
        pixel_length=100.0,
        duration_ms=500.0,
        end_time_ms=1500.0,
    )
    w_obj = WorldObject(id="s_rep5", hit_object=slider)
    anchor_ns = 1_000_000_000_000
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    slide_moves = [i for i in intents if i.cursor_owner == CursorOwnerState.SLIDER_FOLLOW and "slide_" in i.action_id]
    assert len(slide_moves) >= 20

    # Collect sampled progress values
    progresses = [i.metadata.get("progress") for i in slide_moves if i.metadata.get("progress") is not None]

    # With repeat_count=5, turnarounds occur at p = 0.2, 0.4, 0.6, 0.8 and tail at 1.0
    for expected_p in [0.2, 0.4, 0.6, 0.8, 1.0]:
        matching = [p for p in progresses if abs(p - expected_p) < 0.001]
        assert len(matching) >= 1, f"Missing exact turnaround point at p={expected_p}!"

    # Verify turnaround alternating coordinates:
    # p=0.2 (tail): x=200
    # p=0.4 (head): x=100
    # p=0.6 (tail): x=200
    # p=0.8 (head): x=100
    # p=1.0 (tail): x=200
    pts = slider._cached_path
    assert abs(brain._interpolate_slider_path(pts, 0.2, repeat_count=5)[0] - 200.0) < 0.1
    assert abs(brain._interpolate_slider_path(pts, 0.4, repeat_count=5)[0] - 100.0) < 0.1
    assert abs(brain._interpolate_slider_path(pts, 0.6, repeat_count=5)[0] - 200.0) < 0.1
    assert abs(brain._interpolate_slider_path(pts, 0.8, repeat_count=5)[0] - 100.0) < 0.1
    assert abs(brain._interpolate_slider_path(pts, 1.0, repeat_count=5)[0] - 200.0) < 0.1


# ---------------------------------------------------------------------------
# 3. Cursor Ownership Protection & Foreign Move Suppression
# ---------------------------------------------------------------------------

def test_circle_during_active_slider_cannot_hijack_cursor():
    """
    Verify that when a Hit Circle appears immediately after or during an active slider,
    no cursor movement for the Hit Circle is scheduled before the slider safe tail.
    """
    brain = StandardBrain()
    # Slider from 1000ms to 1400ms (duration 400ms, tail at 1400ms, safe tail at 1412ms)
    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=2,
        curve_type=CurveType.LINEAR,
        curve_points=[(200.0, 100.0)],
        repeat_count=1,
        pixel_length=100.0,
        duration_ms=400.0,
        end_time_ms=1400.0,
    )
    # Circle at 1415ms (only 15ms after slider tail, before/around safe tail)
    circle = HitCircle(
        x=400.0,
        y=400.0,
        time_ms=1415.0,
        hit_type=1,
    )
    w_slider = WorldObject(id="s1", hit_object=slider)
    w_circle = WorldObject(id="c1", hit_object=circle)

    anchor_ns = 1_000_000_000_000
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_slider, w_circle], timeline_anchor_ns=anchor_ns)

    # Find circle movements
    circle_moves = [
        i for i in intents
        if i.target_id == "c1" and i.action_type == ActionType.MOUSE_MOVE
    ]

    from src.gameplay.osu.timing.calibration import timing_calibrator
    # Safe tail dispatch time: 1000 + 400 + 12 = 1412ms in calibrated dispatch domain
    safe_tail_dispatch_ns = timing_calibrator.calculate_dispatch_time_ns(1412.0, anchor_ns)

    # Any circle movement MUST NOT occur before safe_tail_dispatch_ns!
    for cm in circle_moves:
        assert cm.execute_at_ns >= safe_tail_dispatch_ns, (
            f"Circle movement {cm.action_id} scheduled at "
            f"{(cm.execute_at_ns - anchor_ns)/1e6:.1f}ms before slider safe tail!"
        )


def test_controller_cursor_ownership_suppresses_foreign_moves():
    """
    Verify CentralInputController drops foreign MOUSE_MOVE intents while
    a slider is actively held down, preventing mouse displacement.
    """
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend)

    now = clock.now_ns()

    # 1. Slider Key Down
    slider_down = ActionIntent(
        action_id="act_s1_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=now,
        deadline_ns=now + 500_000_000,
        target_id="s1",
        key="z",
        metadata={"is_slider": True},
    )
    controller.execute_action(slider_down)
    assert "z" in controller._locked_slider_keys
    assert controller._active_slider_owner == "s1"

    # 2. Foreign MOUSE_MOVE from a Hit Circle arrives while slider is active
    foreign_move = ActionIntent(
        action_id="act_c1_move_1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now + 50_000_000,
        deadline_ns=now + 100_000_000,
        target_id="c1",
        x=500.0,
        y=500.0,
        cursor_owner=CursorOwnerState.CIRCLE_TARGET,
    )
    ok = controller.execute_action(foreign_move)
    assert ok is True
    # Action must be CANCELLED by controller, NOT executed!
    assert foreign_move.status == InputStateStatus.CANCELLED
    assert controller.cursor_pos != (500.0, 500.0)

    # 3. Legitimate SLIDER_FOLLOW MOUSE_MOVE for s1
    slider_move = ActionIntent(
        action_id="act_s1_slide_1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now + 60_000_000,
        deadline_ns=now + 110_000_000,
        target_id="s1",
        x=250.0,
        y=250.0,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
    )
    ok_slide = controller.execute_action(slider_move)
    assert ok_slide is True
    assert slider_move.status == InputStateStatus.SUBMISSION
    assert controller.cursor_pos == (250.0, 250.0)

    # 4. Slider Key Up: safely unlocks slider
    slider_up = ActionIntent(
        action_id="act_s1_up",
        action_type=ActionType.KEY_UP,
        execute_at_ns=now + 200_000_000,
        deadline_ns=now + 250_000_000,
        target_id="s1",
        key="z",
        metadata={"is_slider": True},
    )
    controller.execute_action(slider_up)
    assert "z" not in controller._locked_slider_keys
    assert controller._active_slider_owner is None

    # 5. Subsequent foreign MOUSE_MOVE now executes cleanly
    subsequent_move = ActionIntent(
        action_id="act_c1_move_2",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now + 210_000_000,
        deadline_ns=now + 260_000_000,
        target_id="c1",
        x=500.0,
        y=500.0,
        cursor_owner=CursorOwnerState.CIRCLE_TARGET,
    )
    controller.execute_action(subsequent_move)
    assert subsequent_move.status == InputStateStatus.SUBMISSION
    assert controller.cursor_pos == (500.0, 500.0)


# ---------------------------------------------------------------------------
# 4. Queue Ordering, Priority, and Congestion
# ---------------------------------------------------------------------------

def test_queue_congestion_maintains_slider_integrity():
    """
    Verify CentralInputQueue prioritizes SLIDER_FOLLOW (priority 1) over
    standard moves (priority 2) when execute times are tied.
    """
    queue = CentralInputQueue()
    now = clock.now_ns()

    act_circle = ActionIntent(
        action_id="act_c1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now,
        deadline_ns=now + 100_000_000,
        priority=2,
        cursor_owner=CursorOwnerState.CIRCLE_TARGET,
    )
    act_slider = ActionIntent(
        action_id="act_s1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now,
        deadline_ns=now + 100_000_000,
        priority=1,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
    )

    # Push circle first, then slider
    queue.push(act_circle)
    queue.push(act_slider)

    # Slider MUST pop first due to priority 1 < 2
    popped_1 = queue.pop_due(now)
    assert popped_1 is not None
    assert popped_1.action_id == "act_s1"

    popped_2 = queue.pop_due(now)
    assert popped_2 is not None
    assert popped_2.action_id == "act_c1"


def test_stale_slider_action_does_not_cause_premature_jump():
    """
    Verify that if a slider action experiences execution delay, KEY_UP is
    preserved and pop_due correctly handles stale items.
    """
    queue = CentralInputQueue()
    now = clock.now_ns()

    # Slider slide move that is stale (deadline passed)
    stale_slide = ActionIntent(
        action_id="slide_stale",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now - 100_000_000,
        deadline_ns=now - 10_000_000,
        priority=1,
    )
    # Slider key up that is also overdue
    stale_up = ActionIntent(
        action_id="up_stale",
        action_type=ActionType.KEY_UP,
        execute_at_ns=now - 50_000_000,
        deadline_ns=now - 10_000_000,
        priority=3,
        key="z",
    )

    queue.push(stale_slide)
    queue.push(stale_up)

    # pop_due: stale MOUSE_MOVE is discarded, but KEY_UP is preserved for safety!
    popped = queue.pop_due(now)
    assert popped is not None
    assert popped.action_type == ActionType.KEY_UP


# ---------------------------------------------------------------------------
# 5. Playfield Coordinate Transformation & Screen Offsets
# ---------------------------------------------------------------------------

def test_playfield_scaling_and_window_offsets():
    """
    Verify screen coordinate scaling transforms apply cleanly to all slider
    waypoints when osu! runs in windowed mode with desktop offsets.
    """
    prep = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    prep.update_window_position(offset_x=100, offset_y=50)

    slider = Slider(
        x=256.0,
        y=192.0,
        time_ms=1000.0,
        hit_type=2,
        curve_type=CurveType.LINEAR,
        curve_points=[(356.0, 192.0)],
        repeat_count=1,
        pixel_length=100.0,
        duration_ms=300.0,
        end_time_ms=1300.0,
    )
    w_obj = WorldObject(id="s_offset", hit_object=slider)
    brain = StandardBrain(preprocessor=prep)

    anchor_ns = 1_000_000_000_000
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    slide_moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and "slide_" in i.action_id]
    assert len(slide_moves) >= 5

    # Check that all waypoints have screen coordinates offset by >= window offset
    for m in slide_moves:
        assert m.x > 100.0, f"Waypoint X={m.x} is not properly offset by window_offset_x!"
        assert m.y > 50.0, f"Waypoint Y={m.y} is not properly offset by window_offset_y!"


# ---------------------------------------------------------------------------
# 6. Structured Diagnostics Telemetry Emission
# ---------------------------------------------------------------------------

def test_slider_diagnostics_telemetry_emission():
    """
    Verify that during gameplay app execution, slider_diagnostic events are
    emitted with structured progress, repeat index, and cursor ownership.
    """
    state_store.mutate_domain(
        "mode",
        lambda m: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)

    # Mock telemetry logger
    app.telemetry.log_event = MagicMock()

    now = clock.now_ns()
    app._timeline_anchor_ns = now - int(1000.0 * 1_000_000)

    # Queue a slider follow move
    slide_act = ActionIntent(
        action_id="act_s1_slide_1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=now,
        deadline_ns=now + 50_000_000,
        target_id="s1",
        x=450.0,
        y=350.0,
        priority=1,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
        metadata={"progress": 0.5, "repeat_index": 0},
    )
    app.queue.push(slide_act)

    app.step(simulated_time_ms=1000.0)

    # Verify telemetry.log_event was called with "slider_diagnostic"
    logged_event_types = [call.args[0] for call in app.telemetry.log_event.call_args_list]
    assert "slider_diagnostic" in logged_event_types

    # Find the payload
    diag_call = next(call for call in app.telemetry.log_event.call_args_list if call.args[0] == "slider_diagnostic")
    payload = diag_call.args[1]
    assert payload["target_id"] == "s1"
    assert payload["progress"] == 0.5
    assert payload["repeat_index"] == 0
    assert payload["expected_pos"] == (450.0, 350.0)
    assert payload["cursor_owner"] == "slider_follow"
