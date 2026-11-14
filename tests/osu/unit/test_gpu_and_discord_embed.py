"""
Unit Tests for NVIDIA GPU Acceleration & Modern Discord Embed Reporter
========================================================================
Verifies:
1. GpuAccelerationEngine initializes NVIDIA CUDA Driver (or gracefully falls back),
   reporting accurate VRAM and device status.
2. GPU/Vector-accelerated minimum-jerk trajectory solver generates valid smooth waypoints.
3. Precomputed slider curve waypoints accurately sample path within duration.
4. Discord reporter constructs modern, rich, color-coded embeds:
   - Match start embed (with GPU info & song info)
   - Combo milestone embed (hot pink with live2d quote)
   - Match summary embed with visual accuracy progress bar, rank-based colors, hits breakdown
   - Slapstick fail embed (crimson red)
5. OsuGameplayApp status exposes GPU acceleration telemetry.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.beatmap.models import HitCircle, Slider
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.gpu_engine import GpuAccelerationEngine, gpu_engine
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.integrations.discord_reporter import OsuDiscordReporter, osu_discord_reporter
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


def test_gpu_engine_hardware_status():
    """Verify GPU acceleration engine reports valid hardware info and device status."""
    status = gpu_engine.get_hardware_status()
    assert isinstance(status, dict)
    assert "gpu_accelerated" in status
    assert "device_name" in status
    assert "backend" in status
    assert "free_vram_mb" in status
    assert "total_vram_mb" in status

    # On NVIDIA system with RTX 2060, check that hardware was properly identified
    if status["gpu_accelerated"]:
        assert "GeForce" in status["device_name"] or "RTX" in status["device_name"] or "NVIDIA" in status["device_name"]
        assert status["total_vram_mb"] > 1000.0


def test_gpu_minimum_jerk_trajectory():
    """Verify GPU trajectory solver computes smooth polynomial waypoints."""
    start = (100.0, 100.0)
    target = (500.0, 400.0)
    steps = 5

    waypoints = gpu_engine.compute_minimum_jerk_trajectory(start, target, steps=steps)
    assert len(waypoints) == steps

    # Last waypoint must land on target
    last_x, last_y, last_u = waypoints[-1]
    assert abs(last_x - target[0]) < 1e-4
    assert abs(last_y - target[1]) < 1e-4
    assert abs(last_u - 1.0) < 1e-4

    # Trajectory should monotonically advance progress
    for i in range(len(waypoints) - 1):
        assert waypoints[i][2] < waypoints[i + 1][2]


def test_gpu_precompute_slider_waypoints():
    """Verify slider curve waypoints are sampled smoothly over duration."""
    pts = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0)]
    duration = 500.0  # 500ms

    waypoints = gpu_engine.precompute_slider_waypoints(pts, total_duration_ms=duration, sample_hz=100.0)
    assert len(waypoints) >= 10
    # First waypoint at t=0
    assert waypoints[0][0] == 0.0
    assert waypoints[0][1] == 100.0
    assert waypoints[0][2] == 100.0
    # Final waypoint at t=duration
    assert waypoints[-1][0] == duration


def test_discord_accuracy_bar_rendering():
    """Verify visual emoji progress bar for various accuracy levels."""
    bar_100 = OsuDiscordReporter._render_accuracy_bar(100.0)
    assert "[██████████]" in bar_100
    assert "100.00%" in bar_100

    bar_50 = OsuDiscordReporter._render_accuracy_bar(50.0)
    assert "[█████░░░░░]" in bar_50
    assert "50.00%" in bar_50

    bar_0 = OsuDiscordReporter._render_accuracy_bar(0.0)
    assert "[░░░░░░░░░░]" in bar_0


def test_discord_match_start_embed_builder():
    """Verify match start embed contains hardware info, color, and commentary."""
    reporter = OsuDiscordReporter()
    payload = {
        "type": "start",
        "title": "FREEDOM DiVE",
        "artist": "xi",
        "difficulty": "FOUR DIMENSIONS",
        "mode": "STANDARD",
        "bpm": 222.2,
        "stars": 7.45,
        "gpu_device": "NVIDIA GeForce RTX 2060",
        "free_vram_mb": 5110.0,
        "commentary": "Shiori vào trận rồi nha Ba! ♡",
    }
    embed = reporter._build_embed(payload)
    assert embed["color"] == 0x00E5FF  # Electric Cyan
    assert "FREEDOM DiVE" in embed["description"]
    assert "xi" in embed["description"]
    assert "BPM" in embed["description"]
    assert any("NVIDIA GeForce RTX 2060" in f["value"] for f in embed["fields"])


def test_discord_combo_milestone_embed_builder():
    """Verify combo milestone embed is hot pink and features celebration."""
    reporter = OsuDiscordReporter()
    payload = {
        "type": "milestone",
        "title": "Bad Apple!!",
        "mode": "TAIKO",
        "milestone": 100,
        "accuracy": 99.4,
        "current_combo": 100,
        "commentary": "100 combo đỉnh cao rồi Ba ơi!",
    }
    embed = reporter._build_embed(payload)
    assert embed["color"] == 0xFF4081  # Hot Pink
    assert "100x COMBO" in embed["title"]
    assert any("100x" in f["value"] for f in embed["fields"])
    assert any("99.40%" in f["value"] for f in embed["fields"])


def test_discord_match_summary_embed_rank_colors_and_fail():
    """Verify summary embed selects appropriate color and badge for SS, A, and Failed."""
    reporter = OsuDiscordReporter()

    # 1. SS Rank (Gold)
    p_ss = {
        "type": "summary",
        "title": "Song SS",
        "artist": "Artist",
        "difficulty": "Expert",
        "mode": "STANDARD",
        "rank": "SS",
        "is_failed": False,
        "accuracy": 100.0,
        "max_combo": 500,
        "hits": {"300": 500, "100": 0, "50": 0, "miss": 0},
    }
    e_ss = reporter._build_embed(p_ss)
    assert e_ss["color"] == 0xFFD700
    assert "SS" in e_ss["title"]

    # 2. Failed with Pan / Slapstick (Crimson Red)
    p_fail = {
        "type": "summary",
        "title": "Song Fail",
        "artist": "Artist",
        "difficulty": "Insane",
        "mode": "TAIKO",
        "rank": "F",
        "is_failed": True,
        "accuracy": 45.0,
        "max_combo": 40,
        "hits": {"300": 30, "100": 10, "50": 5, "miss": 8},
    }
    e_fail = reporter._build_embed(p_fail)
    assert e_fail["color"] == 0xFF1744
    assert "FAILED" in e_fail["title"]
    assert "CHẢO" in e_fail["title"]


def test_app_status_exposes_gpu_acceleration():
    """Verify OsuGameplayApp.get_status() includes GPU acceleration telemetry."""
    app = OsuGameplayApp()
    status = app.get_status()
    assert "gpu_accelerated" in status
    assert "gpu_device" in status
    assert "gpu_vram_free_mb" in status


def test_discord_match_start_and_combo_milestone_suppressed():
    """Verify start announcements and combo milestones are suppressed per user request."""
    reporter = OsuDiscordReporter()
    with patch.object(reporter, "_dispatch_async") as mock_dispatch:
        reporter.send_match_start(
            title="Song",
            artist="Artist",
            difficulty="Hard",
            mode="STANDARD",
        )
        reporter.send_combo_milestone(
            title="Song",
            mode="STANDARD",
            milestone=100,
            accuracy=99.0,
            current_combo=100,
        )
        mock_dispatch.assert_not_called()


def test_discord_dispatch_suppressed_in_test_env():
    """Verify _dispatch_async never starts threads or sends HTTP requests when running in pytest."""
    reporter = OsuDiscordReporter()
    with patch("threading.Thread") as mock_thread:
        # PYTEST_CURRENT_TEST is naturally present in os.environ during pytest
        reporter._dispatch_async({"type": "summary", "title": "Test Song"})
        mock_thread.assert_not_called()


def test_song_completed_dispatches_discord_summary():
    """Verify _handle_song_completed triggers Discord summary with is_failed=False and happy emotion."""
    app = OsuGameplayApp()
    app.feedback.metrics.count_300 = 50
    app.feedback.metrics.current_combo = 50
    app.feedback.metrics.max_combo = 50

    with patch.object(osu_discord_reporter, "send_game_summary") as mock_send, \
         patch.object(app, "_emit_emotion_and_thought") as mock_emit:
        app._handle_song_completed()
        assert mock_send.called
        kwargs = mock_send.call_args[1]
        assert kwargs["is_failed"] is False
        assert kwargs["status"] == "completed"
        assert kwargs["emotion"] == "happy"
        assert kwargs["max_combo"] == 50
        assert mock_emit.called
        assert mock_emit.call_args[1]["reason"] == "song_completed"
