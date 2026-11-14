"""
Unit Tests for Mouse Movement, Mouse Buttons, Slider Dragging & Standard Mode Gameplay
"""

import math
import time
import pytest
from unittest.mock import MagicMock, patch, Mock

from src.gameplay.osu.beatmap.models import HitCircle, Slider, Spinner
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    state_store,
)
from src.gameplay.osu.input.action_model import ActionIntent, ActionType
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.planning.validator import ActionValidator
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


def test_mouse_down_up_and_click_lifecycle():
    """Verify CentralInputController dispatches mouse down, up, and timed click."""
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend)

    # 1. Mouse Down
    assert controller.mouse_down("left") is True
    assert "left" in backend.active_mouse_buttons

    # 2. Mouse Up
    assert controller.mouse_up("left") is True
    assert "left" not in backend.active_mouse_buttons

    # 3. Mouse Click (timed tap)
    assert controller.mouse_click("left") is True
    assert "left" in backend.active_mouse_buttons

    # Wait for release worker (12ms tap hold)
    time.sleep(0.040)
    assert "left" not in backend.active_mouse_buttons


def test_stuck_mouse_button_watchdog():
    """Verify watchdog releases mouse button held longer than max_stuck_duration_ms."""
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend, max_stuck_duration_ms=20.0)

    controller.mouse_down("right")
    assert "right" in backend.active_mouse_buttons

    # Simulate passing time
    now_ns = clock.now_ns()
    controller._mouse_down_times["right"] = now_ns - int(50 * 1_000_000)

    stuck_count = controller.check_stuck_keys()
    assert stuck_count >= 1
    assert "right" not in backend.active_mouse_buttons


def test_preprocessor_window_offset_mapping():
    """Verify window client offset shifts osu! coordinates to global screen desktop."""
    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    prep.update_window_position(168, 128)

    # Center of osu playfield (256, 192)
    sx, sy = prep.osu_to_screen(256.0, 192.0)

    # Must be offset by at least window_offset_x (168) and window_offset_y (128)
    assert sx > 168
    assert sy > 128

    # Inverse transform returns original osu coordinates
    osu_x, osu_y = prep.screen_to_osu(sx, sy)
    assert abs(osu_x - 256.0) < 1.0
    assert abs(osu_y - 192.0) < 1.0


def test_standard_brain_slider_dragging_and_waypoints():
    """Verify StandardBrain generates intermediate mouse waypoints and holds mouse/key down."""
    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    brain = StandardBrain(preprocessor=prep, smoothing_steps=3)

    slider = Slider(
        x=100.0,
        y=100.0,
        time_ms=1000.0,
        hit_type=2,  # Slider
        curve_points=[(200.0, 150.0), (300.0, 200.0)],
        repeat_count=1,
        duration_ms=300.0,
        end_time_ms=1300.0,
    )

    w_obj = WorldObject(id="test_slider", hit_object=slider)
    anchor_ns = clock.now_ns()

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    # Verify aiming moves prior to slider head
    pre_moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and "move_" in i.action_id]
    assert len(pre_moves) == 3

    # Verify slider down (both key and mouse)
    down_acts = [i for i in intents if i.action_type in (ActionType.KEY_DOWN, ActionType.MOUSE_DOWN)]
    assert len(down_acts) >= 1
    assert down_acts[0].action_type == ActionType.KEY_DOWN

    # Verify slider dragging intermediate waypoints
    slide_moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE and "slide_" in i.action_id]
    assert len(slide_moves) >= 5, f"Expected slider dragging waypoints, got {len(slide_moves)}"

    # Verify slider up (both key and mouse)
    up_acts = [i for i in intents if i.action_type in (ActionType.KEY_UP, ActionType.MOUSE_UP)]
    assert len(up_acts) >= 1
    assert up_acts[0].action_type == ActionType.KEY_UP


def test_action_validator_mouse_buttons():
    """Verify validator accepts valid mouse buttons and rejects invalid ones."""
    validator = ActionValidator(screen_width=1920, screen_height=1080)

    now = clock.now_ns()
    valid_mdown = ActionIntent("m1", ActionType.MOUSE_DOWN, now, now + 100_000_000, button="left")
    ok, err = validator.validate(valid_mdown)
    assert ok is True

    valid_mclick = ActionIntent("m2", ActionType.MOUSE_CLICK, now, now + 100_000_000, button="right")
    ok, err = validator.validate(valid_mclick)
    assert ok is True

    invalid_btn = ActionIntent("m3", ActionType.MOUSE_DOWN, now, now + 100_000_000, button="wheel_side")
    ok, err = validator.validate(invalid_btn)
    assert ok is False
    assert "Invalid mouse button" in err


def test_auto_aim_disabled_skips_mouse_movement():
    """Verify that when auto_aim_enabled is False, MOUSE_MOVE queue actions are not executed."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.core.state_store import (
        GameMode,
        ModeDetectionStatus,
        ModeState,
        state_store,
    )
    state_store.mutate_domain("mode", lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED), caller="system")
    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app.auto_aim_enabled = False

    app.start(session_id="test_no_aim", run_background_thread=False)
    try:
        now = clock.now_ns()
        # Queue a MOUSE_MOVE and a KEY_TAP
        move_act = ActionIntent("test_move", ActionType.MOUSE_MOVE, now - 1000, now + 1000000, x=500.0, y=500.0)
        key_act = ActionIntent("test_tap", ActionType.KEY_TAP, now - 1000, now + 1000000, key="z")
        app.queue.push(move_act)
        app.queue.push(key_act)

        app.step(simulated_time_ms=100.0)
        # KEY_TAP should be executed, but MOUSE_MOVE must be skipped!
        move_events = [h for h in backend.history if h[0] == "move"]
        key_events = [h for h in backend.history if h[0] == "key_down"]
        assert len(move_events) == 0, "MOUSE_MOVE should have been skipped when auto_aim_enabled=False!"
        assert len(key_events) >= 1, "KEY_TAP should execute normally!"
    finally:
        app.stop()


def test_deadmans_switch_human_motion_yields_control():
    """Verify Win32SendInputBackend yields mouse control when human user moves mouse."""
    from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)

    # Mock user32 methods
    with patch.object(backend, "_get_osu_hwnd", return_value=12345), \
         patch.object(backend._user32, "IsWindow", return_value=True), \
         patch.object(backend._user32, "GetForegroundWindow", return_value=12345), \
         patch.object(backend, "_get_current_cursor_pos", return_value=(650, 650)):

        # Step 1: Shiori moves cursor to (500, 500)
        backend._last_shiori_cursor_pos = (500, 500)
        backend._human_override_until_s = 0.0
        backend._last_move_time_s = 0.0

        # Step 2: Physical cursor at (650, 650) -> Human user moved hand by 212px!
        res = backend.move_cursor(510, 510)
        assert res is False, "Should yield control when human hand is moving mouse!"
        assert backend._human_override_until_s > time.time(), "Cooldown timer should be set!"


def test_standard_hit_circle_and_approach_scale_detection():
    """Verify VisionDetector detects standard hit circles and sorts by approach scale."""
    import cv2
    import numpy as np
    from src.gameplay.osu.perception.detector import VisionDetector
    from src.gameplay.osu.perception.preprocessor import PlayfieldBounds

    # Create synthetic frame with 3 hit circles:
    # Circle A: (200, 200), rim=35, approach circle=35 (scale 1.0)
    # Circle B: (400, 200), rim=35, approach circle=70 (scale 2.0)
    # Circle C: (600, 200), rim=35, approach circle=105 (scale 3.0)
    canvas = np.zeros((480, 800, 3), dtype=np.uint8)

    # Circle A (due now: approach circle collapsed onto rim)
    cv2.circle(canvas, (200, 200), 35, (255, 255, 255), 3)

    # Circle B (approaching: approach circle at r=70)
    cv2.circle(canvas, (400, 200), 35, (255, 255, 255), 3)
    cv2.circle(canvas, (400, 200), 70, (200, 200, 200), 2)

    # Circle C (approaching: approach circle at r=105)
    cv2.circle(canvas, (600, 200), 35, (255, 255, 255), 3)
    cv2.circle(canvas, (600, 200), 105, (200, 200, 200), 2)

    detector = VisionDetector(bounds=PlayfieldBounds(0, 0, 800, 480, 1.0))
    detected = detector.detect_objects(canvas, is_taiko=False)

    assert len(detected) == 3
    # Check strict ascending order by approach scale (Circle A -> Circle B -> Circle C)
    scales = [d.approach_scale for d in detected]
    assert scales == sorted(scales), f"Expected sorted approach scales, got {scales}"
    assert detected[0].approach_scale < detected[1].approach_scale < detected[2].approach_scale
    assert abs(detected[0].x - 200) < 10
    assert abs(detected[1].x - 400) < 10
    assert abs(detected[2].x - 600) < 10


def test_standard_vision_reconciliation_time_ordering():
    """Verify ReconciliationService assigns chronological hit times to circles with varying approach scales."""
    from src.gameplay.osu.world.reconciliation import ReconciliationService, ReconciliationMode
    from src.gameplay.osu.perception.tracker import TrackedObject
    from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
    from src.gameplay.osu.core.state_store import (
        GameMode,
        ModeDetectionStatus,
        ModeState,
        state_store,
    )

    state_store.mutate_domain("mode", lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED), caller="system")
    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    reconciler = ReconciliationService(mode=ReconciliationMode.VISION_FIRST)
    reconciler.world_manager.clear()

    now_ns = clock.now_ns()
    t1 = TrackedObject("trk_1", 200.0, 200.0, 35.0, 1.0, "circle", 0.95, now_ns, now_ns)
    t2 = TrackedObject("trk_2", 400.0, 200.0, 35.0, 2.0, "circle", 0.95, now_ns, now_ns)
    t3 = TrackedObject("trk_3", 600.0, 200.0, 35.0, 3.0, "circle", 0.95, now_ns, now_ns)

    current_time_ms = 1000.0
    results = reconciler.reconcile(current_time_ms=current_time_ms, tracked_objects=[t1, t2, t3], preprocessor=prep, timeline=None)

    assert len(results) == 3
    times = [r.hit_object.time_ms for r in results]
    assert times[0] == 1000.0  # Scale 1.0 is due now!
    assert times[1] > times[0]  # Scale 2.0 is in future!
    assert times[2] > times[1]  # Scale 3.0 is even further in future!


def test_playfield_bounds_rejects_border_hud_and_menu_buttons():
    """Verify VisionDetector strictly ignores circles outside playfield (border HUD & bottom menu)."""
    import cv2
    import numpy as np
    from src.gameplay.osu.perception.detector import VisionDetector
    from src.gameplay.osu.perception.preprocessor import PlayfieldBounds

    # Frame 1280x720:
    # Circle 1: Inside playfield at (500, 360) -> Must be accepted!
    # Circle 2: Far right HUD knob at (1218, 428) -> Must be rejected!
    # Circle 3: Bottom menu button at (983, 700) -> Must be rejected!
    # Circle 4: Top health bar artifact at (200, 30) -> Must be rejected!
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Note 1 (inside playfield with approach circle)
    cv2.circle(canvas, (500, 360), 35, (255, 255, 255), 3)
    cv2.circle(canvas, (500, 360), 70, (200, 200, 200), 2)

    # Circle 2 (far right border)
    cv2.circle(canvas, (1218, 428), 30, (255, 255, 255), 3)

    # Circle 3 (bottom menu area)
    cv2.circle(canvas, (983, 700), 35, (255, 255, 255), 3)

    # Circle 4 (top header bar)
    cv2.circle(canvas, (200, 30), 25, (255, 255, 255), 3)

    detector = VisionDetector(bounds=PlayfieldBounds(0, 0, 1280, 720, 1.0))
    detected = detector.detect_objects(canvas, is_taiko=False)

    # Only Note 1 inside playfield should survive!
    assert len(detected) == 1
    assert abs(detected[0].x - 500) < 10
    assert abs(detected[0].y - 360) < 10
    assert detected[0].has_approach_ring is True


def test_optical_standby_mode_in_song_select_dispatches_zero_inputs():
    """Verify Shiori stays in 0-input standby mode in Song Select before Note #1 spawns."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.beatmap.models import HitCircle, BeatmapData
    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline

    from src.gameplay.osu.core.data_channels import frame_channel
    frame_channel.clear()

    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app.auto_aim_enabled = True
    app.queue.clear()

    # Setup mock beatmap with Note #1 at t=2000ms
    bm = BeatmapData(
        mode=GameMode.STANDARD,
        hit_objects=[HitCircle(x=256.0, y=192.0, time_ms=2000.0, hit_type=1)],
    )
    app._timeline = HitObjectTimeline(bm.hit_objects)
    app.start(session_id="test_standby", run_background_thread=False)

    try:
        # In Song Select / Results screen, Tosu reports is_playing=False -> Standby (0 inputs)
        with patch("src.gameplay.osu.app.tosu_client") as mock_tosu:
            mock_tosu.is_connected = True
            mock_tosu.is_playing = False
            res = app.step()
            assert res == {}
            assert len(backend.history) == 0
    finally:
        app.stop()


def test_dark_intro_screen_with_progress_bar_rejects_bottom_artifacts():
    """Verify VisionDetector produces 0 false notes from bottom progress bar / taskbar in intro screen."""
    import numpy as np
    import cv2
    from src.gameplay.osu.perception.detector import VisionDetector
    from src.gameplay.osu.perception.preprocessor import PlayfieldBounds

    # 1280x720 dark screen with bottom time dots at y=614 and y=607 (outside 4:3 playfield)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(canvas, (861, 614), 15, (255, 255, 255), -1)
    cv2.circle(canvas, (944, 607), 13, (255, 255, 255), -1)

    detector = VisionDetector(bounds=PlayfieldBounds(0, 0, 1280, 720, 1.0))
    res = detector.detect_objects(canvas, is_taiko=False)
    assert len(res) == 0, f"Expected 0 detected objects from bottom bar artifacts, got {len(res)}"


def test_optical_first_hit_slider_holds_down_and_syncs_timeline():
    """Verify first optical hit on a Slider presses and holds mouse/key down, syncing the timeline."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.beatmap.models import Slider, BeatmapData
    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
    from src.gameplay.osu.perception.tracker import TrackedObject

    from src.gameplay.osu.core.state_store import ModeState, ModeDetectionStatus
    state_store.mutate_domain("mode", lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED), caller="system")
    app = OsuGameplayApp()
    app.auto_aim_enabled = True

    # Setup mock beatmap with Note #1 as a Slider at t=1651ms (duration 400ms)
    sl = Slider(
        x=256.0,
        y=192.0,
        time_ms=1651.0,
        hit_type=2,
        duration_ms=400.0,
        curve_points=[(300.0, 200.0), (350.0, 220.0)],
        repeat_count=1,
    )
    bm = BeatmapData(mode=GameMode.STANDARD, hit_objects=[sl])
    app._timeline = HitObjectTimeline(bm.hit_objects)
    app.start(session_id="test_slider_sync", run_background_thread=False)

    try:
        class FakeLiveBackend(MockInputBackend):
            pass
        backend = FakeLiveBackend()
        app.controller.set_backend(backend)

        # Step at slider hit time t=1651ms
        app.step(simulated_time_ms=1651.0)

        # Key down and mouse down must be executed (held down for slider)
        down_actions = [h for h in backend.history if h[0] in ("key_down", "mouse_down")]
        assert len(down_actions) >= 1, f"Expected key_down for slider, got {backend.history}"
        assert down_actions[0][0] == "key_down"
        assert app._actions_dispatched_total >= 1
    finally:
        app.stop()


def test_shared_preprocessor_instance_between_app_and_standard_brain():
    """Verify OsuGameplayApp and StandardBrain share the exact same PlayfieldPreprocessor instance."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.core.state_store import GameMode

    app = OsuGameplayApp()
    brain = app.planner.get_brain(GameMode.STANDARD)
    assert getattr(brain, "preprocessor", None) is app.preprocessor

    # Updating app preprocessor must immediately reflect on brain preprocessor
    app.preprocessor.update_window_position(160, 97)
    app.preprocessor.update_resolution(1280, 720)

    app_coords = app.preprocessor.osu_to_screen(256.0, 192.0)
    brain_coords = brain.preprocessor.osu_to_screen(256.0, 192.0)

    assert app_coords == brain_coords
    assert brain.preprocessor.window_offset_x == 160
    assert brain.preprocessor.window_offset_y == 97


def test_window_offset_coordinate_accuracy_in_windowed_mode():
    """Verify osu! coordinates map with zero drift to windowed client positions."""
    from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor

    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    prep.update_window_position(160, 97)

    # Base scale for 720p: 720 / 480 = 1.5
    assert prep.scale == 1.5
    # Playfield width = 512 * 1.5 = 768. Origin x = (1280 - 768) / 2 = 256.
    # Total screen X for osu x=0 should be window_offset_x (160) + 256 = 416.
    sx_0, sy_0 = prep.osu_to_screen(0.0, 0.0)
    assert sx_0 == 160 + 256

    # Center note (256, 192):
    # sx = 160 + 256 + 256 * 1.5 = 416 + 384 = 800.
    # Center note matches horizontal screen center (160 + 1280 / 2 = 800) exactly!
    sx_c, sy_c = prep.osu_to_screen(256.0, 192.0)
    assert sx_c == 800


def test_dynamic_visual_calibration_eliminates_drift():
    """Verify calibrate_from_visual locks beatmap coordinates to confirmed screen pixels."""
    from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor

    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    prep.update_window_position(160, 97)

    # Suppose user has custom skin or slight visual offset: seen at (1005, 588)
    prep.calibrate_from_visual(visual_x=1005.0, visual_y=588.0, osu_x=386.0, osu_y=265.0)

    # Re-computing coordinate of (386, 265) must now land directly on (1005, 588)
    calibrated_x, calibrated_y = prep.osu_to_screen(386.0, 265.0)
    assert abs(calibrated_x - 1005) <= 1
    assert abs(calibrated_y - 588) <= 1


def test_minimized_window_is_iconic_handling():
    """Verify find_osu_window_region rejects iconic/minimized windows with -32000 coordinates."""
    from unittest.mock import patch
    import ctypes.wintypes

    with patch("src.gameplay.osu.perception.capture.find_osu_window_hwnd", return_value=12345), \
         patch("ctypes.windll.user32.IsIconic", return_value=1):
        from src.gameplay.osu.perception.capture import find_osu_window_region
        reg = find_osu_window_region()
        assert reg is None


def test_mouse_arrival_leads_click_and_priority_inversion_prevented():
    """Verify cursor arrives at hit circle BEFORE key click, and mouse moves take priority in queue."""
    from src.gameplay.osu.input.queue import CentralInputQueue
    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    brain = StandardBrain(preprocessor=prep, smoothing_steps=4)

    circle = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1)
    w_obj = WorldObject(id="circle_lead_test", hit_object=circle)
    anchor_ns = clock.now_ns()

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_obj], timeline_anchor_ns=anchor_ns)

    move_intents = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE]
    click_intents = [i for i in intents if i.action_type == ActionType.KEY_TAP]

    assert len(move_intents) == 4
    assert len(click_intents) == 1

    final_move = move_intents[-1]
    click = click_intents[0]

    # Verification 1: Final mouse move must arrive BEFORE the click dispatch time!
    assert final_move.execute_at_ns < click.execute_at_ns, "Cursor must arrive before key is pressed!"
    lead_ms = (click.execute_at_ns - final_move.execute_at_ns) / 1_000_000.0
    assert 6.0 <= lead_ms <= 24.0, f"Expected lead time between 6ms and 24ms, got {lead_ms}ms"

    # Verification 2: Mouse move priority must be strictly higher (lower numerical value) than key tap
    assert final_move.priority < click.priority, "MOUSE_MOVE must have higher priority than KEY_TAP to avoid clicking at stale coordinates!"

    # Verification 3: In CentralInputQueue, when both are due, MOUSE_MOVE is popped before KEY_TAP
    queue = CentralInputQueue()
    queue.push(click)
    queue.push(final_move)

    first_popped = queue.pop_due(click.execute_at_ns)
    second_popped = queue.pop_due(click.execute_at_ns)

    assert first_popped.action_type == ActionType.MOUSE_MOVE, "Queue must pop MOUSE_MOVE before KEY_TAP!"
    assert second_popped.action_type == ActionType.KEY_TAP


def test_tier1_target_latching_prevents_aim_jitter():
    """Verify StandardBrain planning produces smooth monotonic minimum-jerk waypoints without jitter."""
    from src.gameplay.osu.brains.standard_brain import StandardBrain
    from src.gameplay.osu.world.object_lifecycle import WorldObject
    from src.gameplay.osu.beatmap.models import HitCircle

    brain = StandardBrain(smoothing_steps=5)
    c1 = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1)
    w1 = WorldObject(id="c1", hit_object=c1)
    anchor = clock.now_ns()

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w1], timeline_anchor_ns=anchor)
    moves = [i for i in intents if i.action_type == ActionType.MOUSE_MOVE]
    assert len(moves) == 5

    # Coordinates must progress monotonically towards target (100, 100)
    dists = [((m.x - 100.0)**2 + (m.y - 100.0)**2)**0.5 for m in moves]
    for i in range(len(dists) - 1):
        assert dists[i] >= dists[i+1], "Aim path must progress monotonically without jitter!" 





def test_standard_brain_button_none_and_keyboard_only_dispatch():
    """Verify StandardBrain generates button=None for all hit actions (circles, sliders, spinners) to prevent mouse drag disruption."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = clock.now_ns()

    circle = WorldObject("c1", HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=1))
    slider = WorldObject("s1", Slider(x=300.0, y=300.0, time_ms=1500.0, hit_type=2, duration_ms=400.0))
    spinner = WorldObject("sp1", Spinner(x=256.0, y=192.0, time_ms=2200.0, hit_type=8, duration_ms=500.0))

    intents = brain.plan_actions(current_time_ms=800.0, objects=[circle, slider, spinner], timeline_anchor_ns=anchor_ns)

    hit_actions = [i for i in intents if i.action_type in (ActionType.KEY_TAP, ActionType.KEY_DOWN, ActionType.KEY_UP)]
    assert len(hit_actions) >= 4, "Must generate hit actions for circle, slider start/end, and spinner start/end"

    for action in hit_actions:
        assert action.button is None, f"Action {action.action_id} should have button=None (got {action.button}) to prevent mouse chording conflicts"
        assert action.key in ("z", "x"), f"Action {action.action_id} must use keyboard keys 'z' or 'x'"


def test_controller_slider_key_never_released_by_prior_tap_worker():
    """Verify CentralInputController protects locked slider keys from being prematurely released by tap timers or alternating taps."""
    backend = Mock()
    backend.key_down.return_value = True
    backend.key_up.return_value = True

    controller = CentralInputController(backend=backend)

    now = clock.now_ns()

    # 1. Simulate an earlier KEY_TAP that scheduled a release timer for 'z' in 10ms
    tap_intent = ActionIntent(
        action_id="tap_early",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=now,
        deadline_ns=now + 50_000_000,
        key="z",
        metadata={"tap_hold_ms": 10.0},
    )
    controller.execute_action(tap_intent)
    assert "z" in controller._key_down_times
    assert len(controller._release_heap) == 1

    # 2. 2ms later, a Slider starts on 'z' (KEY_DOWN)
    slider_down_intent = ActionIntent(
        action_id="slider_start",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=now + 2_000_000,
        deadline_ns=now + 52_000_000,
        key="z",
    )
    controller.execute_action(slider_down_intent)

    # Verification: KEY_DOWN must lock 'z' as a slider key and purge pending release timers
    assert "z" in controller._locked_slider_keys
    assert len(controller._release_heap) == 0, "KEY_DOWN must purge pending release timers for the same key"

    # 3. Simulate another tap arriving while slider is locked: it should redirect to 'x' without touching 'z'
    mid_slider_tap = ActionIntent(
        action_id="tap_during_slider",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=now + 50_000_000,
        deadline_ns=now + 100_000_000,
        key="z",
        metadata={"tap_hold_ms": 10.0},
    )
    controller.execute_action(mid_slider_tap)
    # Target redirected to alternate key 'x'
    assert mid_slider_tap.key == "x"
    assert "z" in controller._locked_slider_keys

    # 4. End slider explicitly with KEY_UP
    slider_up_intent = ActionIntent(
        action_id="slider_end",
        action_type=ActionType.KEY_UP,
        execute_at_ns=now + 200_000_000,
        deadline_ns=now + 250_000_000,
        key="z",
    )
    controller.execute_action(slider_up_intent)
    assert "z" not in controller._locked_slider_keys


def test_trajectory_continuity_across_horizon_without_future_aim_jerk():
    """Verify that simulated cursor trajectory is continuous across horizon objects and does not jerk to future positions."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep, smoothing_steps=4)
    anchor_ns = clock.now_ns()

    # Map with 3 circles: c1 at (100, 100), c2 at (200, 200), c3 at (450, 450)
    c1 = WorldObject("c1", HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1))
    c2 = WorldObject("c2", HitCircle(x=200.0, y=200.0, time_ms=1300.0, hit_type=1))
    c3 = WorldObject("c3", HitCircle(x=450.0, y=450.0, time_ms=2000.0, hit_type=1))

    # Cycle 1: Horizon contains all 3 objects.
    intents_cycle1 = brain.plan_actions(current_time_ms=800.0, objects=[c1, c2, c3], timeline_anchor_ns=anchor_ns)
    moves_c1 = [i for i in intents_cycle1 if i.target_id == "c1" and i.action_type == ActionType.MOUSE_MOVE]
    moves_c2 = [i for i in intents_cycle1 if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE]

    # c2 must start near c1's target position (200, 200 screen coords), NOT from c3's (450, 450) position!
    c1_screen_x, c1_screen_y = prep.osu_to_screen(100.0, 100.0)
    c2_start_move = moves_c2[0]
    # First step towards c2 must be close to c1's screen position
    dist_from_c1 = math.hypot(c2_start_move.x - c1_screen_x, c2_start_move.y - c1_screen_y)
    c3_screen_x, c3_screen_y = prep.osu_to_screen(450.0, 450.0)
    dist_from_c3 = math.hypot(c2_start_move.x - c3_screen_x, c2_start_move.y - c3_screen_y)

    assert dist_from_c1 < dist_from_c3, "c2 must plan trajectory starting from c1, not from future c3!"

    # Now simulate c1 having been scheduled/committed in Cycle 2:
    c1.lifecycle = ObjectLifecycleState.ACTION_SCHEDULED
    intents_cycle2 = brain.plan_actions(current_time_ms=900.0, objects=[c1, c2, c3], timeline_anchor_ns=anchor_ns)
    moves_c2_cycle2 = [i for i in intents_cycle2 if i.target_id == "c2" and i.action_type == ActionType.MOUSE_MOVE]

    assert len(moves_c2_cycle2) == 4
    # Trajectory of c2 in cycle 2 must STILL start near c1's end position!
    dist_cycle2 = math.hypot(moves_c2_cycle2[0].x - c1_screen_x, moves_c2_cycle2[0].y - c1_screen_y)
    assert dist_cycle2 < dist_from_c3, "When c1 is ACTION_SCHEDULED, c2 must still smoothly start from c1's position without aim teleportation!"


def test_aim_disjoint_intervals_prevents_slider_break_and_circle_misaim():
    """Verify that subsequent objects never dispatch mouse moves before the previous circle or slider has finished."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep)
    anchor_ns = clock.now_ns()

    c0 = WorldObject("c0", HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1))
    s1 = WorldObject("s1", Slider(x=200.0, y=200.0, time_ms=1150.0, hit_type=2, duration_ms=200.0))
    s2 = WorldObject("s2", Slider(x=350.0, y=350.0, time_ms=1450.0, hit_type=2, duration_ms=150.0))

    intents = brain.plan_actions(current_time_ms=900.0, objects=[c0, s1, s2], timeline_anchor_ns=anchor_ns)

    c0_tap = [i for i in intents if i.target_id == "c0" and i.action_type == ActionType.KEY_TAP][0]
    s1_moves = [i for i in intents if i.target_id == "s1" and i.action_type == ActionType.MOUSE_MOVE and "slide" not in i.action_id]
    s1_up = [i for i in intents if i.target_id == "s1" and i.action_type == ActionType.KEY_UP][0]
    s2_moves = [i for i in intents if i.target_id == "s2" and i.action_type == ActionType.MOUSE_MOVE and "slide" not in i.action_id]

    # 1. s1 aim moves must NEVER start before c0 has completed its hit
    for m in s1_moves:
        assert m.execute_at_ns >= c0_tap.execute_at_ns, "s1 must not move mouse away before c0 is clicked!"

    # 2. s2 aim moves must NEVER start before s1 slider has completed (KEY_UP)
    for m in s2_moves:
        assert m.execute_at_ns >= s1_up.execute_at_ns, "s2 must not move mouse away before s1 slider is finished!"


def test_stream_pacing_reduces_aim_steps_for_rapid_notes():
    """Verify that rapid stream notes (<110ms apart) generate compact snappy aim steps."""
    prep = PlayfieldPreprocessor(1280, 720)
    brain = StandardBrain(preprocessor=prep, smoothing_steps=4)
    anchor_ns = clock.now_ns()

    # 3 notes in rapid succession (stream at 160 BPM: 93ms apart)
    n1 = WorldObject("n1", HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=1))
    n2 = WorldObject("n2", HitCircle(x=220.0, y=210.0, time_ms=1093.0, hit_type=1))
    n3 = WorldObject("n3", HitCircle(x=240.0, y=220.0, time_ms=1186.0, hit_type=1))

    intents = brain.plan_actions(current_time_ms=950.0, objects=[n1, n2, n3], timeline_anchor_ns=anchor_ns)
    moves_n2 = [i for i in intents if i.target_id == "n2" and i.action_type == ActionType.MOUSE_MOVE]

    # For 93ms interval, aim steps must be compact (<=2 steps) to prevent input queue congestion
    assert 1 <= len(moves_n2) <= 2
    # Final step must arrive on target coordinates
    n2_sx, n2_sy = prep.osu_to_screen(220.0, 210.0)
    assert moves_n2[-1].x == float(n2_sx)
    assert moves_n2[-1].y == float(n2_sy)


def test_deadman_switch_macro_vs_micro_jitter():
    """Verify Win32SendInputBackend ignores micro-movements (<120px) but yields on macro human moves."""
    from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)

    with patch.object(backend, "_get_osu_hwnd", return_value=12345), \
         patch.object(backend._user32, "IsWindow", return_value=True), \
         patch.object(backend._user32, "GetForegroundWindow", return_value=12345):

        # Scenario 1: Micro-jitter / fast jumps within 40px -> MUST NOT YIELD!
        backend._last_shiori_cursor_pos = (500, 500)
        backend._human_override_until_s = 0.0
        with patch.object(backend, "_get_current_cursor_pos", return_value=(535, 520)):
            res = backend.move_cursor(540, 525)
            assert res is True, "Micro-movement (<120px) should never trigger human override!"
            assert backend._human_override_until_s == 0.0

        # Scenario 2: Human user deliberately jerks mouse across screen (distance = 200px) -> YIELDS!
        with patch.object(backend, "_get_current_cursor_pos", return_value=(700, 700)):
            res2 = backend.move_cursor(540, 525)
            assert res2 is False, "Deliberate macro-movement (>120px) must yield control to human!"
            assert backend._human_override_until_s > time.time()


def test_slider_curve_circumscribed_circle_sampling():
    """Verify that sample_perfect_circle follows genuine circular arc."""
    from src.gameplay.osu.math.slider_curves import sample_perfect_circle
    # Semicircle from (100, 200) through (200, 100) to (300, 200) (center at 200, 200, R=100)
    pts = sample_perfect_circle((100.0, 200.0), (200.0, 100.0), (300.0, 200.0), pixel_length=314.159, num_points=25)
    assert len(pts) >= 20
    # Midpoint of sampled curve should be exactly at the apex (200, 100)
    mid_pt = pts[len(pts) // 2]
    assert abs(mid_pt[0] - 200.0) < 1.0
    assert abs(mid_pt[1] - 100.0) < 1.0


def test_slider_curve_bezier_sampling():
    """Verify that sample_bezier_segments computes smooth De Casteljau curve."""
    from src.gameplay.osu.math.slider_curves import sample_bezier_segments
    ctrls = [(100.0, 100.0), (200.0, 300.0), (300.0, 100.0)]
    pts = sample_bezier_segments(ctrls, num_points_per_seg=21)
    assert len(pts) >= 15
    # Apex of quadratic Bezier with controls (100,100), (200,300), (300,100) at t=0.5 is (200, 200)
    mid_pt = pts[len(pts) // 2]
    assert abs(mid_pt[0] - 200.0) < 1.0
    assert abs(mid_pt[1] - 200.0) < 1.0


def test_bpm_adaptive_timing_calibration():
    """Verify that timing_calibrator adapts latency based on tempo."""
    from src.gameplay.osu.timing.calibration import timing_calibrator
    timing_calibrator.calibrate_for_bpm(190.0)
    assert timing_calibrator.system_latency_ms == 9.0

    timing_calibrator.calibrate_for_bpm(150.0)
    assert timing_calibrator.system_latency_ms == 12.0


def test_tosu_synchronization_damping_and_seek():
    """Verify that Tosu micro-jitter uses smooth damping while seek/skip triggers sync without wiping metrics."""
    from src.gameplay.osu.app import OsuGameplayApp

    app = OsuGameplayApp()
    app.feedback.metrics.current_combo = 150
    app.feedback.metrics.accuracy = 98.5

    # 1. Test _session_reset with reset_metrics=False preserves metrics
    app._session_reset(reason="gameplay-resume", reset_metrics=False)
    assert app.feedback.metrics.current_combo == 150
    assert app.feedback.metrics.accuracy == 98.5

    # 2. Test _session_reset with reset_metrics=True resets metrics
    app._session_reset(reason="retry", reset_metrics=True)
    assert app.feedback.metrics.current_combo == 0
    assert app.feedback.metrics.accuracy == 100.0


def test_pause_resume_recovery_preserves_metrics():
    """Verify that unpause/resume in lazer preserves gameplay metrics and re-arms tosu sync."""
    from src.gameplay.osu.app import OsuGameplayApp

    app = OsuGameplayApp()
    app._is_running = True
    app.controller._backend = MagicMock()
    app.feedback.metrics.current_combo = 42
    app.feedback.metrics.accuracy = 95.0
    app._lazer_is_paused = True
    app._pause_start_ns = clock.now_ns() - 1_000_000_000  # 1s pause
    app._tosu_anchor_synced = True

    # Simulate resume detection within the same map and player screen
    app._lazer_resumed = True
    app._last_lazer_working_map = "TestMap"
    app._last_seen_player_screen = "SoloPlayer"
    app._current_lazer_player_screen = "SoloPlayer"

    with patch.object(app, "_check_lazer_gameplay_state", return_value=(True, "TestMap")):
        with patch("ctypes.windll.user32.IsWindow", return_value=1):
            app._cached_osu_hwnd = 12345
            with patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=0):
                app.step(simulated_time_ms=None)

    # After resume, metrics should NOT be wiped to 0
    assert app.feedback.metrics.current_combo == 42
    assert app.feedback.metrics.accuracy == 95.0
    # And tosu anchor is flagged for immediate lock when audio starts
    assert app._tosu_anchor_synced is False
    assert app._lazer_resumed is False



