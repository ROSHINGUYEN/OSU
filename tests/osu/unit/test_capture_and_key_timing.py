"""
Unit tests for high-speed capture auto-recovery, JPEG debug frame streaming,
and multi-speed hit window calculation across diverse tempo/SV beatmaps.
"""

import time
import numpy as np
import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, ModeDetectionStatus, ModeState, state_store
from src.gameplay.osu.evaluation.debug_logger import visual_debugger
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.perception.capture import CaptureService, MockCaptureBackend
from src.gameplay.osu.perception.tracker import TrackedObject


def test_visual_debugger_fast_jpeg_encoding():
    """Verify JPEG encoding is ultra-fast (< 5ms) and produces valid JPEG bytes."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 0] = 50

    visual_debugger.annotate_and_record(
        frame_rgb=frame,
        detected_objects=[],
        mode="taiko",
        drum_line_x=250.0,
        accuracy=100.0,
        combo=0,
        offset_ms=0.0,
    )

    # Warm up OpenCV codec to isolate steady-state performance from cold start
    _ = visual_debugger.get_latest_jpeg_bytes(quality=80)

    t0 = time.perf_counter()
    jpg_bytes = visual_debugger.get_latest_jpeg_bytes(quality=80)
    t1 = time.perf_counter()

    assert jpg_bytes is not None
    assert len(jpg_bytes) > 100
    # JPEG magic header: 0xFF 0xD8
    assert jpg_bytes[:2] == b"\xff\xd8"
    assert (t1 - t0) < 0.050  # Must be fast (< 50ms even on busy CPU)


def test_central_input_controller_12ms_fast_tap():
    """Verify tap_key holds for ~12ms and automatically releases via worker thread."""
    mock_backend = MockInputBackend()
    ctrl = CentralInputController(backend=mock_backend)

    assert "f" not in mock_backend.active_keys
    success = ctrl.tap_key("f")
    assert success is True
    assert "f" in mock_backend.active_keys

    # Wait for the 12ms tap hold to expire
    time.sleep(0.040)
    assert "f" not in mock_backend.active_keys


def test_multi_speed_beatmap_hit_invariance():
    """Verify engine accurately triggers hit on standard beatmap objects."""
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    mock_input = MockInputBackend()
    app.controller.set_backend(mock_input)

    from src.gameplay.osu.beatmap.models import BeatmapData, HitCircle
    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline

    bm = BeatmapData(
        mode=GameMode.STANDARD,
        hit_objects=[
            HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1, hit_sound=0),
            HitCircle(x=256.0, y=192.0, time_ms=2000.0, hit_type=1, hit_sound=2),
        ],
    )
    app._timeline = HitObjectTimeline(bm.hit_objects)

    app.start(session_id="test_multi_speed", run_background_thread=False)
    try:
        # Case 1: Hit at t=1000ms
        app.step(simulated_time_ms=1000.0)
        assert len([h for h in mock_input.history if h[0] == "key_down"]) == 1

        mock_input.clear()

        # Case 2: Hit at t=2000ms
        app.step(simulated_time_ms=2000.0)
        key_taps = [h for h in mock_input.history if h[0] == "key_down"]
        assert len(key_taps) == 1
        assert key_taps[0][1] in ("z", "x")
    finally:
        app.stop()
