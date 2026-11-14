"""
Unit Tests for Taiko High-BPM Stream Alternation, Big Note Dual Hits,
Standard Slider Preservation on Overlapping Objects, and Anti-Stuck Key Re-Tap.
"""

import pytest

from src.gameplay.osu.beatmap.models import HitCircle, Slider
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode
from src.gameplay.osu.input.action_model import ActionType
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject



def test_standard_slider_preservation_when_circle_appears():
    """Verify that while a Slider is active, an overlapping Circle picks the alternate key and zero mouse clicks are emitted."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    # Slider from 1000ms to 1400ms (duration 400ms)
    sl = Slider(x=100, y=100, time_ms=1000.0, hit_type=2, duration_ms=400.0, end_time_ms=1400.0)
    w_slider = WorldObject("sl1", sl)

    # Circle appearing at 1200ms while slider is being held
    circle = HitCircle(x=300, y=300, time_ms=1200.0, hit_type=1)
    w_circle = WorldObject("c1", circle)

    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_slider, w_circle], timeline_anchor_ns=anchor_ns)

    # 1. Zero mouse clicks or mouse down/up (preserves mouse tracking and prevents breaking slider)
    mouse_clicks = [i for i in intents if i.action_type in (ActionType.MOUSE_CLICK, ActionType.MOUSE_DOWN, ActionType.MOUSE_UP)]
    assert len(mouse_clicks) == 0, f"Expected 0 mouse clicks/downs for standard mode, got {mouse_clicks}"

    # 2. Slider down key
    slider_down = [i for i in intents if i.target_id == "sl1" and i.action_type == ActionType.KEY_DOWN][0]
    slider_key = slider_down.key
    assert slider_key in ("z", "x")

    # 3. Circle key while slider is active MUST NOT be the same key as the slider!
    circle_tap = [i for i in intents if i.target_id == "c1" and i.action_type == ActionType.KEY_TAP][0]
    assert circle_tap.key != slider_key, f"Circle used same key '{circle_tap.key}' as active slider key '{slider_key}'!"

    # 4. Slider up key matches slider down key
    slider_up = [i for i in intents if i.target_id == "sl1" and i.action_type == ActionType.KEY_UP][0]
    assert slider_up.key == slider_key


def test_anti_stuck_re_tap_forces_key_up_before_new_down():
    """Verify that calling tap_key on an already-held key forces key_up immediately so both strokes register."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)

    # First tap
    res1 = ctrl.tap_key("f")
    assert res1 is True
    assert "f" in ctrl._key_down_times

    # Immediate second tap before release worker runs
    res2 = ctrl.tap_key("f")
    assert res2 is True
    assert "f" in ctrl._key_down_times

    # Inspect backend history: key_down -> key_up (forced release) -> key_down (clean stroke)
    key_events = [h[0] for h in backend.history if h[1] == "f"]
    assert key_events[:3] == ["key_down", "key_up", "key_down"], f"Expected forced key_up before second key_down, got {key_events}"
    assert ctrl._tap_hold_ns == int(0.012 * 1_000_000_000), f"Expected 12ms tap hold, got {ctrl._tap_hold_ns}"



def test_controller_slider_locked_key_protection():
    """Verify CentralInputController protects locked slider keys and remaps colliding taps to alternate key."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)

    # 1. Slider begins on 'z'
    ctrl.key_down("z")
    assert "z" in ctrl._locked_slider_keys

    # 2. While slider is held on 'z', a tap request arrives for 'z'
    res = ctrl.tap_key("z")
    assert res is True

    # 3. Tap was safely remapped to alternate key 'x' so slider 'z' is never released!
    assert "x" in ctrl._key_down_times
    assert "z" in ctrl._locked_slider_keys
    # Backend history must have: key_down('z') then tap_key redirected to 'x' (key_down 'x')
    downs = [h[1] for h in backend.history if h[0] == "key_down"]
    assert downs == ["z", "x"], f"Expected ['z', 'x'], got {downs}"

    # 4. Slider ends
    ctrl.key_up("z")
    assert "z" not in ctrl._locked_slider_keys


def test_standard_brain_rapid_stream_alternation():
    """Verify consecutive circles within 220ms alternate keys z -> x -> z -> x."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    objs = [
        WorldObject(f"c_{i}", HitCircle(x=200 + i * 10, y=200, time_ms=1000.0 + i * 60.0, hit_type=1))
        for i in range(4)
    ]

    intents = brain.plan_actions(current_time_ms=900.0, objects=objs, timeline_anchor_ns=anchor_ns)
    taps = [i.key for i in intents if i.action_type == ActionType.KEY_TAP]
    assert taps == ["z", "x", "z", "x"], f"Expected strict alternation ['z', 'x', 'z', 'x'], got {taps}"


def test_slider_overlapping_circle_suppresses_mouse_and_protects_slider_key():
    """Verify overlapping circle during active slider picks alternate key, suppresses mouse click, and doesn't break slider."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    # Slider from 1000ms to 1500ms
    slider = Slider(x=100.0, y=100.0, time_ms=1000.0, hit_type=2, duration_ms=500.0)
    w_slider = WorldObject("s1", slider, lifecycle=ObjectLifecycleState.APPROACHING)

    # First cycle: plan slider at t=950ms
    slider_intents = brain.plan_actions(current_time_ms=950.0, objects=[w_slider], timeline_anchor_ns=anchor_ns)
    slider_down = [i for i in slider_intents if i.action_type == ActionType.KEY_DOWN][0]
    slider_key = slider_down.key
    assert slider_key in ["z", "x"]

    # Overlapping circle at 1250ms (while slider is still active until 1500ms)
    circle = HitCircle(x=150.0, y=150.0, time_ms=1250.0, hit_type=1)
    w_circle = WorldObject("c1", circle, lifecycle=ObjectLifecycleState.APPROACHING)

    circle_intents = brain.plan_actions(current_time_ms=1200.0, objects=[w_circle], timeline_anchor_ns=anchor_ns)
    circle_tap = [i for i in circle_intents if i.action_type == ActionType.KEY_TAP][0]

    # Verification 1: Circle MUST pick the opposite key so slider key is never interrupted
    expected_alt = "x" if slider_key == "z" else "z"
    assert circle_tap.key == expected_alt, f"Expected alternate key {expected_alt}, got {circle_tap.key}"

    # Verification 2: Circle MUST NOT click mouse buttons while slider is active (prevents breaking slider mouse hold)
    assert circle_tap.button is None, f"Expected button to be None during active slider, got {circle_tap.button}"


def test_stacked_circles_in_high_bpm_stream_snappy_and_safe_tap_hold():
    """Verify stacked circles (<6px) in high-BPM stream (<70ms) use safe 22ms+ tap hold and snappy steps."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep, smoothing_steps=4)
    anchor_ns = 1_000_000_000

    # Two stacked circles at (256, 192) spaced 50ms apart (~240 BPM 1/4 stream)
    c1 = WorldObject("c1", HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1))
    c2 = WorldObject("c2", HitCircle(x=256.0, y=192.0, time_ms=1050.0, hit_type=1))

    # Plan first circle
    intents1 = brain.plan_actions(current_time_ms=900.0, objects=[c1], timeline_anchor_ns=anchor_ns)
    # Plan second circle during stream (50ms delta)
    intents2 = brain.plan_actions(current_time_ms=1000.0, objects=[c2], timeline_anchor_ns=anchor_ns)

    tap2 = [i for i in intents2 if i.action_type == ActionType.KEY_TAP][0]
    moves2 = [i for i in intents2 if i.action_type == ActionType.MOUSE_MOVE]

    # Verification 1: Tap hold is safely between 20ms and 35ms (22.5ms for 50ms stream) to avoid osu!lazer frame drop
    assert 20.0 <= tap2.metadata.get("tap_hold_ms") <= 35.0, f"Expected safe tap hold between 20-35ms, got {tap2.metadata.get('tap_hold_ms')}"

    # Verification 2: Stacked note in high-BPM stream uses 1 snappy step instead of 4 intermediate steps
    assert len(moves2) == 1, f"Expected 1 snappy move step for stacked stream note, got {len(moves2)}"


def test_controller_key_down_and_tap_decouples_mouse_movement():
    """Verify executing KEY_DOWN or KEY_TAP never moves cursor or mutates cursor_pos (cursor decoupled)."""
    from src.gameplay.osu.input.action_model import ActionIntent
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)
    ctrl.auto_aim_enabled = True

    # 1. KEY_TAP with coordinates (450, 320)
    tap_action = ActionIntent(
        action_id="test_tap",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=1_000_000,
        deadline_ns=2_000_000,
        mode=GameMode.STANDARD,
        key="z",
        x=450.0,
        y=320.0,
    )
    ok_tap = ctrl.execute_action(tap_action)
    assert ok_tap is True
    # Mouse cursor must remain untouched by key actions (no cursor jerk/conflict)
    assert ctrl.cursor_pos == (0.0, 0.0)
    assert not any(h[0] == "move_cursor" for h in backend.history)
    assert backend.history[-1][:2] == ("key_down", "z")

    # 2. KEY_DOWN with coordinates (600, 480) for slider start
    down_action = ActionIntent(
        action_id="test_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=3_000_000,
        deadline_ns=4_000_000,
        mode=GameMode.STANDARD,
        key="x",
        x=600.0,
        y=480.0,
    )
    ok_down = ctrl.execute_action(down_action)
    assert ok_down is True
    assert ctrl.cursor_pos == (0.0, 0.0)
    assert not any(h[0] == "move_cursor" for h in backend.history)
    assert "x" in ctrl._locked_slider_keys
    assert backend.history[-1][:2] == ("key_down", "x")


def test_stream_leading_into_slider_smooth_transition_and_lock():
    """Verify stacked stream circles leading immediately into a slider alternate keys and lock slider key."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = 1_000_000_000

    # 2 circles spaced 94ms apart (160 BPM 1/4), leading directly into a 400ms slider
    c1 = WorldObject("c1", HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=1))
    c2 = WorldObject("c2", HitCircle(x=205.0, y=205.0, time_ms=1094.0, hit_type=1))
    sl = WorldObject("sl1", Slider(x=210.0, y=210.0, time_ms=1188.0, hit_type=2, duration_ms=400.0))

    intents = brain.plan_actions(current_time_ms=900.0, objects=[c1, c2, sl], timeline_anchor_ns=anchor_ns)

    c1_tap = next(i for i in intents if i.target_id == "c1" and i.action_type == ActionType.KEY_TAP)
    c2_tap = next(i for i in intents if i.target_id == "c2" and i.action_type == ActionType.KEY_TAP)
    sl_down = next(i for i in intents if i.target_id == "sl1" and i.action_type == ActionType.KEY_DOWN)
    sl_up = next(i for i in intents if i.target_id == "sl1" and i.action_type == ActionType.KEY_UP)

    # Key alternation: c1('z') -> c2('x') -> sl('z')
    assert c1_tap.key == "z"
    assert c2_tap.key == "x"
    assert sl_down.key == "z"
    assert sl_up.key == "z"

    # Tap hold duration is safely within 22-35ms
    assert 22.0 <= c1_tap.metadata.get("tap_hold_ms") <= 35.0
    assert 22.0 <= c2_tap.metadata.get("tap_hold_ms") <= 35.0

    # Slider head has valid screen coordinates matching target
    exp_x, exp_y = prep.osu_to_screen(210.0, 210.0)
    assert int(sl_down.x) == int(exp_x)
    assert int(sl_down.y) == int(exp_y)


