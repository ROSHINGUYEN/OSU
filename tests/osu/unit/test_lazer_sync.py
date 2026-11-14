"""
Unit tests for osu!lazer runtime state synchronization and beatmap discovery.
"""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend


def test_lazer_gameplay_state_detection():
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "osu" / "logs"
        logs_dir.mkdir(parents=True)
        log_file = logs_dir / "123456.runtime.log"

        # 1. Menu state
        log_file.write_text(
            "2026-10-26 08:52:52 [verbose]: entered MainMenu#290\n"
            "2026-10-26 08:58:17 [verbose]: entered SoloSongSelect#886\n"
            "2026-10-26 08:58:20 [verbose]: Game-wide working beatmap updated to Artist - Song (Creator) [Easy]\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"APPDATA": tmp_dir}):
            is_playing, working_map = app._check_lazer_gameplay_state(time.time() + 1.0)
            assert is_playing is False
            assert working_map == "Artist - Song (Creator) [Easy]"

            # 2. Transition into gameplay (SoloPlayer)
            log_file.write_text(
                "2026-10-26 08:58:17 [verbose]: entered SoloSongSelect#886\n"
                "2026-10-26 08:58:20 [verbose]: Game-wide working beatmap updated to Artist - Song (Creator) [Hard]\n"
                "2026-10-26 08:58:25 [verbose]: entered SoloSongSelect+PlayerLoader#101\n"
                "2026-10-26 08:58:27 [verbose]: entered SoloPlayer#555\n",
                encoding="utf-8",
            )
            is_playing, working_map = app._check_lazer_gameplay_state(time.time() + 2.0)
            assert is_playing is True
            assert working_map == "Artist - Song (Creator) [Hard]"

            # 3. Transition into Results/Menu
            log_file.write_text(
                "2026-10-26 08:58:27 [verbose]: entered SoloPlayer#555\n"
                "2026-10-26 09:00:10 [verbose]: entered Results#777\n",
                encoding="utf-8",
            )
            is_playing, working_map = app._check_lazer_gameplay_state(time.time() + 3.0)
            assert is_playing is False


def test_lazer_pause_and_resume_state_tracking():
    """Verify pause (StopGameplayClock) and resume (StartGameplayClock) are accurately detected."""
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())

    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "osu" / "logs"
        logs_dir.mkdir(parents=True)
        log_file = logs_dir / "pause_test.runtime.log"

        # 1. Player playing normally
        log_file.write_text(
            "2026-10-27 12:00:00 [verbose]: entered SoloPlayer#999\n"
            "2026-10-27 12:00:01 [verbose]: StartGameplayClock\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"APPDATA": tmp_dir}):
            is_playing, _ = app._check_lazer_gameplay_state(time.time() + 1.0)
            assert is_playing is True
            assert app._lazer_is_paused is False

            # 2. Player hits ESC (pause)
            log_file.write_text(
                "2026-10-27 12:00:00 [verbose]: entered SoloPlayer#999\n"
                "2026-10-27 12:00:01 [verbose]: StartGameplayClock\n"
                "2026-10-27 12:00:15 [verbose]: StopGameplayClock\n",
                encoding="utf-8",
            )
            is_playing, _ = app._check_lazer_gameplay_state(time.time() + 2.0)
            assert is_playing is False
            assert app._lazer_is_paused is True
            assert app._pause_start_ns > 0

            # 3. Player hits Continue (resume)
            log_file.write_text(
                "2026-10-27 12:00:00 [verbose]: entered SoloPlayer#999\n"
                "2026-10-27 12:00:01 [verbose]: StartGameplayClock\n"
                "2026-10-27 12:00:15 [verbose]: StopGameplayClock\n"
                "2026-10-27 12:00:25 [verbose]: StartGameplayClock\n",
                encoding="utf-8",
            )
            is_playing, _ = app._check_lazer_gameplay_state(time.time() + 3.0)
            assert is_playing is True
            assert app._lazer_is_paused is False
            assert app._lazer_resumed is True


def test_lazer_resume_adjusts_timeline_anchor():
    """Verify that when lazer resumes from pause, the timeline anchor shifts by pause duration."""
    class DummyLiveBackend(MockInputBackend):
        pass

    app = OsuGameplayApp()
    app.controller.set_backend(DummyLiveBackend())
    app._is_running = True

    from src.gameplay.osu.core.clock import clock
    now_ns = clock.now_ns()

    # Initial state: 10 seconds into song, anchor set to (now_ns - 10s)
    app._timeline_anchor_ns = now_ns - 10_000_000_000
    app._last_seen_player_screen = "SoloPlayer#999"
    app._current_lazer_player_screen = "SoloPlayer#999"
    app._last_lazer_working_map = "Test Map"

    # Simulate paused 3.5 seconds ago
    pause_duration_ns = 3_500_000_000
    app._pause_start_ns = now_ns - pause_duration_ns
    app._lazer_is_paused = False
    app._lazer_resumed = True

    with patch.object(app, "_check_lazer_gameplay_state", return_value=(True, "Test Map")):
        with patch("src.gameplay.osu.perception.capture.find_osu_window_hwnd", return_value=12345):
            with patch("ctypes.windll.user32.IsWindow", return_value=1):
                with patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=0):
                    app.step(simulated_time_ms=None)

    # Timeline anchor should have shifted forward by pause duration
    expected_new_anchor = (now_ns - 10_000_000_000) + pause_duration_ns
    assert abs(app._timeline_anchor_ns - expected_new_anchor) < 100_000_000  # Within 100ms
    assert app._lazer_resumed is False
    assert len(app.queue._heap) == 0


def test_pause_does_not_trigger_song_completed_nor_flush_queue():
    """Verify that pausing does NOT falsely complete the match or wipe future queued actions."""
    from unittest.mock import MagicMock
    from src.gameplay.osu.input.action_model import ActionIntent, ActionType
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    from src.gameplay.osu.evaluation.feedback import FeedbackEvaluator

    app = OsuGameplayApp(feedback=FeedbackEvaluator())
    app._is_running = True
    app._was_active_play = True
    app._actions_dispatched_total = 20
    app._lazer_is_paused = True
    app._last_lazer_working_map = "TestMap"
    app._last_seen_player_screen = "SoloPlayer"
    app._current_lazer_player_screen = "SoloPlayer"
    # In-game clock is frozen at 1250.0ms (paused)
    app._last_tosu_advancing_time = 1250.0

    # Put a scheduled action into queue
    test_action = ActionIntent(
        action_id="future_tap",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=100_000_000,
        deadline_ns=150_000_000,
        key="z",
    )
    app.queue.push(test_action)
    assert len(app.queue._heap) == 1

    orig_conn = tosu_client._is_connected
    orig_play = tosu_client.is_playing
    tosu_client._is_connected = True
    tosu_client.is_playing = True
    try:
        with patch.object(app, "_check_lazer_gameplay_state", return_value=(False, "TestMap")), \
             patch.object(tosu_client, "time_ms", 1250.0), \
             patch.object(app, "_handle_song_completed") as mock_complete:
            res = app.step(simulated_time_ms=None)
            assert res == {}
            # Song completed must NEVER be called on pause!
            mock_complete.assert_not_called()
            # Queue must remain intact, NOT flushed!
            assert len(app.queue._heap) == 1
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_play


def test_tosu_advancing_clock_clears_pause_and_syncs_all_metrics():
    """Verify advancing tosu audio clock clears pause and syncs max combo and hit counts."""
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    from src.gameplay.osu.evaluation.feedback import FeedbackEvaluator

    app = OsuGameplayApp(feedback=FeedbackEvaluator())
    app._is_running = True
    app._lazer_is_paused = True
    app._last_tosu_advancing_time = 1000.0

    orig_conn = tosu_client._is_connected
    orig_play = tosu_client.is_playing
    tosu_client._is_connected = True
    tosu_client.is_playing = True
    try:
        with patch.object(tosu_client, "time_ms", 1250.0), \
             patch.object(tosu_client, "current_combo", 16), \
             patch.object(tosu_client, "max_combo", 25), \
             patch.object(tosu_client, "accuracy", 98.74), \
             patch.object(tosu_client, "hits_300", 30), \
             patch.object(tosu_client, "hits_100", 2), \
             patch.object(tosu_client, "hits_50", 1), \
             patch.object(tosu_client, "hits_miss", 0), \
             patch.object(tosu_client, "get_interpolated_time_ms", return_value=1250.0), \
             patch.object(app, "_check_lazer_gameplay_state", return_value=(False, None)), \
             patch("ctypes.windll.user32.IsWindow", return_value=1):
            app._cached_osu_hwnd = 12345
            app.step(simulated_time_ms=None)
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_play

    # Pause must be cleared
    assert app._lazer_is_paused is False
    assert app._lazer_resumed is True
    # Metrics must be synchronized
    assert app.feedback.metrics.current_combo == 16
    assert app.feedback.metrics.max_combo == 25
    assert app.feedback.metrics.accuracy == 98.74
    assert app.feedback.metrics.count_300 == 30
    assert app.feedback.metrics.count_100 == 2
    assert app.feedback.metrics.count_50 == 1
    assert app.feedback.metrics.count_miss == 0
    assert app.feedback.metrics.total_hits == 33


def test_input_controller_remapped_key_released_cleanly_on_key_up():
    """Verify that when KEY_DOWN remaps a key (e.g. z -> x), KEY_UP releases that exact remapped key."""
    from src.gameplay.osu.input.action_model import ActionIntent, ActionType
    from src.gameplay.osu.input.controller import CentralInputController

    ctrl = CentralInputController()
    mock_b = MockInputBackend()
    ctrl.set_backend(mock_b)

    # Lock key 'z' with an existing slider
    ctrl._locked_slider_keys.add("z")

    # Target 123 requests 'z'
    down_action = ActionIntent(
        action_id="act_123_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=0,
        deadline_ns=1000,
        target_id="circle_123",
        key="z",
        created_at_ns=0,
    )
    ctrl.execute_action(down_action)

    # Since 'z' was locked, down_action was remapped to 'x'
    assert down_action.key == "x"
    assert "x" in ctrl._key_down_times
    assert "x" in ctrl._locked_slider_keys

    # Matching KEY_UP arrives with original key="z"
    up_action = ActionIntent(
        action_id="act_123_up",
        action_type=ActionType.KEY_UP,
        execute_at_ns=0,
        deadline_ns=1000,
        target_id="circle_123",
        key="z",
        created_at_ns=0,
    )
    ctrl.execute_action(up_action)

    # KEY_UP must have resolved to 'x' and released it
    assert up_action.key == "x"
    assert "x" not in ctrl._key_down_times
    assert "x" not in ctrl._locked_slider_keys


def test_stale_stop_clock_before_screen_entry_does_not_pause():
    """Verify that StopGameplayClock occurring before entering the current player screen does NOT trigger false pause."""
    import tempfile
    app = OsuGameplayApp()
    with tempfile.TemporaryDirectory() as tmp_dir:
        logs_dir = Path(tmp_dir) / "osu" / "logs"
        logs_dir.mkdir(parents=True)
        log_file = logs_dir / "123456.runtime.log"

        # Scenario: Old match paused (StopGameplayClock), then player returned to SongSelect, then entered SoloPlayer
        log_file.write_text(
            "2026-10-26 08:50:00 [verbose]: StopGameplayClock\n"
            "2026-10-26 08:52:17 [verbose]: entered SoloSongSelect#886\n"
            "2026-10-26 08:58:27 [verbose]: entered SoloPlayer#555\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"APPDATA": tmp_dir}):
            is_playing, _ = app._check_lazer_gameplay_state(time.time() + 1.0)
            # StopGameplayClock was BEFORE entering SoloPlayer -> must NOT be treated as paused!
            assert is_playing is True
            assert app._lazer_is_paused is False


def test_intro_countdown_leadin_not_wiped_by_tosu_zero():
    """Verify that during intro lead-in countdown (e.g. -1335ms), Tosu reporting 0.0ms does not wipe the anchor."""
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    from src.gameplay.osu.core.clock import clock
    app = OsuGameplayApp()
    app._is_running = True

    now_ns = clock.now_ns()
    # Entered SoloPlayer with -1335ms lead-in
    leadin_ms = -1335.0
    expected_anchor = now_ns - int(leadin_ms * 1_000_000)
    app._timeline_anchor_ns = expected_anchor
    app._timeline_synced_from_optics = True
    app._tosu_anchor_synced = False
    app._last_seen_player_screen = "SoloPlayer#123"
    app._current_lazer_player_screen = "SoloPlayer#123"
    app._last_lazer_working_map = "Test Map"

    with patch.object(tosu_client, "_is_connected", True), \
         patch.object(tosu_client, "is_playing", True), \
         patch.object(tosu_client, "time_ms", 0.0), \
         patch.object(tosu_client, "get_interpolated_time_ms", return_value=0.0), \
         patch.object(app, "_check_lazer_gameplay_state", return_value=(True, "Test Map")), \
         patch("src.gameplay.osu.perception.capture.find_osu_window_hwnd", return_value=12345), \
         patch("ctypes.windll.user32.IsWindow", return_value=1), \
         patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=0):

        # Execute step during leadin countdown (Tosu reports 0.0ms, but song audio hasn't started)
        app.step(simulated_time_ms=None)

        # Timeline anchor must NOT have been wiped to now_ns!
        assert app._timeline_anchor_ns == expected_anchor
        assert app._tosu_anchor_synced is False  # Must still wait for positive audio tick!


def test_intro_countdown_transitions_to_positive_tosu_time_cleanly():
    """Verify that when Tosu transitions to positive time (> 0.0ms), timeline locks to audio clock."""
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    from src.gameplay.osu.core.clock import clock
    app = OsuGameplayApp()
    app._is_running = True

    now_ns = clock.now_ns()
    leadin_ms = -1335.0
    expected_anchor = now_ns - int(leadin_ms * 1_000_000)
    app._timeline_anchor_ns = expected_anchor
    app._timeline_synced_from_optics = True
    app._tosu_anchor_synced = False
    app._last_seen_player_screen = "SoloPlayer#123"
    app._current_lazer_player_screen = "SoloPlayer#123"
    app._last_lazer_working_map = "Test Map"

    # Step 1: In countdown (0.0ms)
    with patch.object(tosu_client, "_is_connected", True), \
         patch.object(tosu_client, "is_playing", True), \
         patch.object(tosu_client, "time_ms", 0.0), \
         patch.object(tosu_client, "get_interpolated_time_ms", return_value=0.0), \
         patch.object(app, "_check_lazer_gameplay_state", return_value=(True, "Test Map")), \
         patch("src.gameplay.osu.perception.capture.find_osu_window_hwnd", return_value=12345), \
         patch("ctypes.windll.user32.IsWindow", return_value=1), \
         patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=0):

        app.step(simulated_time_ms=None)
        assert app._timeline_anchor_ns == expected_anchor
        assert app._tosu_anchor_synced is False

        # Step 2: Song audio starts, Tosu reports 15.0ms
        with patch.object(tosu_client, "time_ms", 15.0), \
             patch.object(tosu_client, "get_interpolated_time_ms", return_value=15.0):
            app.step(simulated_time_ms=None)
            assert app._tosu_anchor_synced is True
            # The anchor should now be locked to (now_ns - 15ms)
            step2_now_ns = clock.now_ns()
            assert abs(app._timeline_anchor_ns - (step2_now_ns - int(15.0 * 1_000_000))) < 500_000_000
