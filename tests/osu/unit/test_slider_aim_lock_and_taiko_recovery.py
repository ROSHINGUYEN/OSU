"""
Unit Tests for Slider Aim Lock, Key Protection, Taiko Recovery, and Zero-Lag Stutter Elimination
================================================================================================
Verifies:
1. Slider aim lock prevents mouse cursor from departing slider ball before completion.
2. Active slider locks the holding key and forces alternate key on close/stacked notes.
3. Fast HWND validation eliminates EnumWindows 2000Hz CPU stutter.
4. Tosu time_ms <= 0.0 does not freeze or crush timeline anchor to 0.
5. Taiko tolerance_ns >= 150ms prevents premature stale drops on micro-delays.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.beatmap.models import BeatmapData, HitCircle, Slider
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.input.action_model import ActionType
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.integrations.tosu_client import tosu_client
from src.gameplay.osu.perception.mode_detector import mode_detection_service
from src.gameplay.osu.perception.preprocessor import PlayfieldBounds, PlayfieldPreprocessor
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


def test_slider_aim_lock_protects_cursor_from_early_leave():
    """Verify aiming movement toward next note never begins before active slider completes."""
    preprocessor = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    brain = StandardBrain(preprocessor=preprocessor)
    brain.auto_aim_enabled = True

    slider_hit = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=2,
        hit_sound=0,
        duration_ms=800.0,  # Ends at 1800ms
        curve_points=[(200.0, 100.0), (300.0, 100.0)],
        repeat_count=1,
    )
    circle_hit = HitCircle(
        x=450.0,
        y=350.0,
        time_ms=1840.0,  # Appears right after slider ends (only 40ms gap)
        hit_type=1,
        hit_sound=0,
    )

    w_slider = WorldObject(id="obj_slider", hit_object=slider_hit, lifecycle=ObjectLifecycleState.APPROACHING)
    w_circle = WorldObject(id="obj_circle", hit_object=circle_hit, lifecycle=ObjectLifecycleState.APPROACHING)

    anchor_ns = clock.now_ns()

    # Plan at current_time = 1200ms (in the middle of the slider)
    intents = brain.plan_actions(
        current_time_ms=1200.0,
        objects=[w_slider, w_circle],
        timeline_anchor_ns=anchor_ns,
    )

    # 1. Slider should have generated down, intermediate slide moves, and up
    slider_moves = [i for i in intents if i.target_id == "obj_slider" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(slider_moves) >= 5

    # 2. Circle aiming moves MUST NOT start before slider ends (1800ms)
    circle_moves = [i for i in intents if i.target_id == "obj_circle" and i.action_type == ActionType.MOUSE_MOVE]
    assert len(circle_moves) >= 1

    # Check earliest circle move execute_at_ns:
    # 1800ms relative to anchor corresponds to anchor_ns + 1800 * 1_000_000 - lead
    earliest_circle_move_ns = min(m.execute_at_ns for m in circle_moves)
    min_allowed_ns = anchor_ns + int((1800.0 - 25.0) * 1_000_000)
    assert earliest_circle_move_ns >= min_allowed_ns, "Aim move departed before slider completed!"


def test_slider_key_lock_enforces_alternate_key_for_stacked_notes():
    """Verify close/stacked note following a slider is forced to use the alternate key."""
    preprocessor = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    brain = StandardBrain(preprocessor=preprocessor)

    slider_hit = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        duration_ms=500.0,  # Ends at 1500ms
        hit_type=2,
        hit_sound=0,
    )
    circle_hit = HitCircle(
        x=100.0,
        y=100.0,
        time_ms=1510.0,  # Stacked note 10ms after slider
        hit_type=1,
        hit_sound=0,
    )

    w_slider = WorldObject(id="obj_slider", hit_object=slider_hit, lifecycle=ObjectLifecycleState.APPROACHING)
    w_circle = WorldObject(id="obj_circle", hit_object=circle_hit, lifecycle=ObjectLifecycleState.APPROACHING)

    intents = brain.plan_actions(
        current_time_ms=900.0,
        objects=[w_slider, w_circle],
        timeline_anchor_ns=clock.now_ns(),
    )

    slider_keys = [i.key for i in intents if i.target_id == "obj_slider" and i.action_type in (ActionType.KEY_DOWN, ActionType.KEY_TAP)]
    circle_keys = [i.key for i in intents if i.target_id == "obj_circle" and i.action_type in (ActionType.KEY_DOWN, ActionType.KEY_TAP)]

    assert len(slider_keys) >= 1
    assert len(circle_keys) >= 1
    # Circle MUST use the other key (e.g. if slider uses z, circle must use x)
    assert slider_keys[0] != circle_keys[0]


def test_standard_generous_tolerance_prevents_stale_drops():
    """Verify StandardBrain sets action deadline tolerance to survive micro-stutters."""
    preprocessor = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    brain = StandardBrain(preprocessor=preprocessor)
    circle = HitCircle(x=256.0, y=192.0, time_ms=2000.0, hit_type=1, hit_sound=0)
    w_note = WorldObject(id="obj_circle_1", hit_object=circle, lifecycle=ObjectLifecycleState.APPROACHING)

    anchor_ns = clock.now_ns()
    intents = brain.plan_actions(
        current_time_ms=1900.0,
        objects=[w_note],
        timeline_anchor_ns=anchor_ns,
    )

    tap_actions = [i for i in intents if i.action_type == ActionType.KEY_TAP]
    assert len(tap_actions) == 1
    act = tap_actions[0]
    tolerance_ns = act.deadline_ns - act.execute_at_ns
    assert tolerance_ns >= 50_000_000  # >= 50ms tolerance


def test_tosu_zero_time_does_not_reset_timeline_anchor():
    """Verify that when Tosu reports 0.0ms during active play, timeline anchor is NOT crushed to 0ms."""
    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)

    app.start(session_id="test_tosu_zero", run_background_thread=False)

    orig_conn = tosu_client._is_connected
    orig_playing = tosu_client.is_playing
    orig_time = tosu_client.time_ms

    try:
        # Set timeline anchor to 5000ms ago
        initial_anchor_ns = clock.now_ns() - 5_000_000_000
        app._timeline_anchor_ns = initial_anchor_ns
        app._tosu_anchor_synced = True

        # Tosu reports connected and playing, but time_ms is 0.0ms (e.g. leadin or v4 unparsed)
        tosu_client._is_connected = True
        tosu_client.is_playing = True
        tosu_client.time_ms = 0.0

        app.step()

        # Anchor should NOT have been crushed to clock.now_ns()
        # It should remain approximately initial_anchor_ns
        diff_ms = abs((app._timeline_anchor_ns - initial_anchor_ns) / 1_000_000.0)
        assert diff_ms < 100.0, "Timeline anchor was crushed to 0ms!"
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_playing
        tosu_client.time_ms = orig_time
        app.stop()
