"""
Unit Tests for Buzz Slider Smoothing, Tail Dwell Calibration Alignment, and Deadman Resilience.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.gameplay.osu.beatmap.models import Slider, HitCircle, HitObjectFlag, CurveType
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
from src.gameplay.osu.input.action_model import ActionType, CursorOwnerState
from src.gameplay.osu.planning.planner import WorldObject
from src.gameplay.osu.timing.calibration import timing_calibrator


def test_buzz_slider_smoothing_reduces_step_explosion():
    """Verify that buzz sliders (short length and fast spans) do not spam excessive steps."""
    brain = StandardBrain()
    brain.auto_aim_enabled = True

    # Buzz slider matching Rock The House: len=25.0, repeats=14, duration=600ms (span=42.8ms)
    buzz_slider = Slider(
        x=256.0,
        y=192.0,
        time_ms=1000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(256.0, 192.0), (281.0, 192.0)],
        repeat_count=14,
        pixel_length=25.0,
        duration_ms=600.0,
    )
    buzz_obj = WorldObject(id="buzz_1", hit_object=buzz_slider)

    intents = brain.plan_actions(500.0, [buzz_obj], timeline_anchor_ns=1_000_000_000)
    move_intents = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and "slide_" in i.action_id]

    # Without buzz smoothing, 14 repeats * 4 steps = 56 steps (+ dwell = 57).
    # With buzz smoothing (span_steps = 1 or 2), total steps <= 28 (+ dwell = 29).
    assert len(move_intents) <= 29
    assert len(move_intents) >= 14


def test_slider_tail_dwell_and_key_up_calibration_alignment():
    """Verify that slide_dwell and KEY_UP are both dispatched using timing_calibrator and dwell past end_t_ms."""
    brain = StandardBrain()
    brain.auto_aim_enabled = True

    s = Slider(
        x=100.0,
        y=100.0,
        time_ms=2000.0,
        hit_type=HitObjectFlag.SLIDER,
        curve_type=CurveType.LINEAR,
        curve_points=[(100.0, 100.0), (200.0, 100.0)],
        repeat_count=1,
        pixel_length=100.0,
        duration_ms=400.0,  # end_t_ms = 2400.0
    )
    s_obj = WorldObject(id="s1", hit_object=s)

    # Next object 100ms later
    c = HitCircle(
        x=300.0,
        y=300.0,
        time_ms=2500.0,
        hit_type=HitObjectFlag.CIRCLE,
    )
    c_obj = WorldObject(id="c2", hit_object=c)

    intents = brain.plan_actions(1500.0, [s_obj, c_obj], timeline_anchor_ns=1_000_000_000)

    dwell_intent = [i for i in intents if i.action_id == "act_s1_slide_dwell"][0]
    up_intent = [i for i in intents if i.action_id == "act_s1_up"][0]
    c_move_intent = [i for i in intents if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE][0]

    # Verify dwell is at end_t_ms (2400.0) and up is at safe_tail_ms (2412.0ms) with lead-time calibration
    expected_safe_tail_ms = 2412.0
    expected_up_ns = timing_calibrator.calculate_dispatch_time_ns(expected_safe_tail_ms, 1_000_000_000)
    expected_dwell_ns = timing_calibrator.calculate_dispatch_time_ns(2400.0, 1_000_000_000)

    assert dwell_intent.execute_at_ns == expected_dwell_ns
    assert up_intent.execute_at_ns == expected_up_ns

    # Verify next circle's cursor move does NOT depart before end_t_ms
    assert c_move_intent.execute_at_ns >= expected_dwell_ns


def test_deadman_switch_jump_resilience_and_corner_override():
    """Verify that deliberate corner panic triggers deadman switch."""
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    backend.enable_deadman_switch = True
    backend._last_shiori_cursor_pos = (707, 406)
    backend._last_move_time_s = time.time()  # Active aiming

    # Corner escape (2, 2) triggers human override
    with patch.object(backend, "_get_current_cursor_pos", return_value=(2, 2)), \
         patch.object(backend, "_is_osu_active", return_value=True):

        success = backend.move_cursor(500, 400)
        assert success is False
        assert backend._human_override_until_s > time.time()


def test_cross_screen_jump_step_quantization():
    """Verify that large cross-screen jumps with normal time (>135ms) use multi-step minimum-jerk trajectory."""
    brain = StandardBrain()
    brain.auto_aim_enabled = True

    brain._cursor_x = 420.0
    brain._cursor_y = 100.0

    circle = HitCircle(
        x=50.0,
        y=350.0,
        time_ms=1160.0,
        hit_type=HitObjectFlag.CIRCLE,
    )
    c_obj = WorldObject(id="cross_jump", hit_object=circle)

    intents = brain.plan_actions(1000.0, [c_obj], timeline_anchor_ns=1_000_000_000)
    moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and i.target_id == "cross_jump"]

    # Must NOT collapse into 1 step teleportation
    assert len(moves) >= 3, f"Expected at least 3 steps for large cross-screen jump, got {len(moves)}"
    # Final step must reach exact target
    prep = brain.preprocessor
    target_sx, target_sy = prep.osu_to_screen(50.0, 350.0)
    assert moves[-1].x == target_sx
    assert moves[-1].y == target_sy
    # Key tap must occur AFTER the final move lands with positive lead
    key_tap = [i for i in intents if i.action_type == ActionType.KEY_TAP and i.target_id == "cross_jump"][0]
    lead_time_ns = key_tap.execute_at_ns - moves[-1].execute_at_ns
    assert lead_time_ns >= 10_000_000, f"Expected lead time >= 10ms, got {lead_time_ns / 1e6:.1f}ms"

