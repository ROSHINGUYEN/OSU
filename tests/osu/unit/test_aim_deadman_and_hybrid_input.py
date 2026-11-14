"""
Unit Tests for Aim Deadman Switch, Hybrid Mouse Input, Zero Vision GUI Pop-up, and Unranked Map Sync.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
from src.gameplay.osu.integrations.tosu_client import TosuClient


def test_deadman_switch_startup_corner_does_not_deadlock():
    """Verify that if cursor starts at (0, 0) or near corner on startup, deadman switch does NOT deadlock."""
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    backend.enable_deadman_switch = True
    backend._last_shiori_cursor_pos = None

    # Simulate physical mouse returning (0, 0)
    with patch.object(backend, "_get_current_cursor_pos", return_value=(0, 0)), \
         patch.object(backend, "_is_osu_active", return_value=True), \
         patch.object(backend._user32, "SetCursorPos", return_value=1), \
         patch.object(backend._user32, "SendInput", return_value=1):

        # Initial move
        success = backend.move_cursor(500, 400)
        assert success is True
        assert backend._human_override_until_s == 0.0
        assert backend._last_shiori_cursor_pos == (500, 400)


def test_deadman_switch_deliberate_corner_escape_triggers_safety():
    """Verify that deliberate corner jerk from active playfield triggers 0.5s human override."""
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    backend.enable_deadman_switch = True
    backend._last_shiori_cursor_pos = (600, 500)  # Shiori was actively aiming inside screen

    now = time.time()
    # User deliberately jerks mouse into corner (2, 2)
    with patch.object(backend, "_get_current_cursor_pos", return_value=(2, 2)), \
         patch.object(backend, "_is_osu_active", return_value=True):

        success = backend.move_cursor(700, 500)
        assert success is False
        assert backend._human_override_until_s >= now + 0.45


def test_deadman_switch_song_opening_and_release_all():
    """Verify release_all clears stale state and opening note move does NOT false-trigger deadman switch."""
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    backend.enable_deadman_switch = True
    backend._last_shiori_cursor_pos = (707, 406)  # Stale pos from previous match
    backend._last_move_time_s = time.time() - 2.0  # 2 seconds ago (inactive between matches)

    with patch.object(backend, "_get_current_cursor_pos", return_value=(964, 491)), \
         patch.object(backend, "_is_osu_active", return_value=True), \
         patch.object(backend._user32, "SetCursorPos", return_value=1), \
         patch.object(backend._user32, "SendInput", return_value=1):

        # Scenario 1: Inactivity auto-syncs reference position, move succeeds without false lockout
        success = backend.move_cursor(605, 588)
        assert success is True
        assert backend._human_override_until_s == 0.0

        # Scenario 2: release_all resets cursor history and override cooldown
        backend.release_all()
        assert backend._last_shiori_cursor_pos is None
        assert backend._human_override_until_s == 0.0
        assert backend._last_move_time_s == 0.0


def test_shiori_commander_osu_play_has_no_vision_gui_spawn():
    """Verify that !osu play does NOT spawn shiori_vision_gui.py popup."""
    from src.commands.shiori_commander import ShioriCommander
    commander = ShioriCommander()

    with patch("src.gameplay.osu.app.osu_app.start", return_value=True), \
         patch("subprocess.Popen") as mock_popen:

        success, msg = commander.process_command("!osu play", is_roshi=True)
        assert success is True
        assert ("Memory-Driven" in msg) or ("games_companion" in msg)
        assert "128x128" not in msg

        # Ensure no vision GUI process was spawned
        if mock_popen.called:
            for call in mock_popen.call_args_list:
                args = call[0][0] if call[0] else []
                assert not any("shiori_vision_gui.py" in str(arg) for arg in args)


def test_tosu_client_supports_unranked_beatmaps_with_zero_id():
    """Verify tosu_client fetches and caches beatmaps even when beatmap_id == 0 (unranked/custom)."""
    client = TosuClient()
    client.beatmap_id = 0
    client.beatmap_title = "YUC'e - Future Cake [Extra]"
    client.folder_name = "Future Cake"
    client.file_name = "YUC'e - Future Cake (Roshi) [Extra].osu"

    mock_osu_content = "osu file format v14\n[HitObjects]\n256,192,1000,1,0"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = mock_osu_content.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        content = client.fetch_active_beatmap_content()
        assert content is not None
        assert "[HitObjects]" in content
        assert client._cached_map_key == f"0:Future Cake:YUC'e - Future Cake (Roshi) [Extra].osu:YUC'e - Future Cake [Extra]"


def test_virtual_desktop_sendinput_normalization():
    """Verify Win32SendInputBackend normalizes coordinates using Virtual Desktop (multi-monitor)."""
    backend = Win32SendInputBackend(screen_width=1680, screen_height=1050)
    backend.enable_deadman_switch = False

    # Simulate multi-monitor setup: left=-768, top=0, width=2448, height=1613
    def mock_get_system_metrics(idx):
        metrics = {0: 1680, 1: 1050, 76: -768, 77: 0, 78: 2448, 79: 1613}
        return metrics.get(idx, 0)

    called_send_input = []
    def mock_send_input(n, p_inp, sz):
        called_send_input.append(n)
        return 1

    with patch.object(backend._user32, "GetSystemMetrics", side_effect=mock_get_system_metrics), \
         patch.object(backend, "_is_osu_active", return_value=True), \
         patch.object(backend._user32, "SetCursorPos", return_value=1), \
         patch.object(backend._user32, "SendInput", side_effect=mock_send_input):

        # Move to (840, 501)
        res = backend.move_cursor(840, 501)
        assert res is True
        assert len(called_send_input) == 1


def test_memory_driven_bpm_adaptive_latency():
    """Verify that in Memory-Driven mode, BPM calibration sets 1.0-2.0ms latency instead of 12ms."""
    from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator
    calibrator = AdaptiveTimingCalibrator()

    # Memory mode fast BPM (>175) -> 1.0ms
    calibrator.calibrate_for_bpm(190.0, is_memory_mode=True)
    assert calibrator.system_latency_ms == 1.0

    # Memory mode normal BPM (<=175) -> 2.0ms
    calibrator.calibrate_for_bpm(150.0, is_memory_mode=True)
    assert calibrator.system_latency_ms == 2.0

    # Optics mode fast BPM (>175) -> 9.0ms
    calibrator.calibrate_for_bpm(190.0, is_memory_mode=False)
    assert calibrator.system_latency_ms == 9.0


def test_osu_mcp_server_calibrate_command():
    """Verify that osu_mcp_server handles direct calibrate and latency tuning commands."""
    from scripts.osu_mcp_server import OsuMCPServer
    from src.gameplay.osu.timing.calibration import timing_calibrator

    server = OsuMCPServer()

    # Tune latency directly
    res = server.handle_send_command({"command": "calibrate", "args": "lead 2.5"})
    assert res.get("success") is True
    assert timing_calibrator.system_latency_ms == 2.5

    # Tune offset directly
    res2 = server.handle_send_command({"command": "calibrate", "args": "0.0"})
    assert res2.get("success") is True
