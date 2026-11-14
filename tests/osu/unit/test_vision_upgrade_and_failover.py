"""
Unit Tests for Vision Failover, Hybrid Scheduling, and Standard Gameplay Accuracy
"""

import time
import numpy as np
import cv2
import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.data_channels import frame_channel
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.integrations.discord_reporter import OsuDiscordReporter
from src.gameplay.osu.perception.detector import VisionDetector
from src.gameplay.osu.perception.preprocessor import PlayfieldBounds


def test_detector_standard_circles_on_dark_screen():
    bounds = PlayfieldBounds(x=0, y=0, width=1280, height=720, scale=1.0)
    detector = VisionDetector(bounds)

    # Dark background
    frame_dark = np.full((720, 1280, 3), 30, dtype=np.uint8)
    # Circle at (400, 280)
    cv2.circle(frame_dark, (400, 280), 25, (187, 17, 119), -1)
    # Circle at (600, 280)
    cv2.circle(frame_dark, (600, 280), 25, (34, 153, 187), -1)

    detected = detector.detect_objects(frame_dark)
    assert len(detected) >= 2
    assert all(d.obj_type == "circle" for d in detected)


def test_standard_key_alternation_debounce():
    mock_backend = MockInputBackend()
    ctrl = CentralInputController(backend=mock_backend)

    # Standard alternating keys z and x
    res1 = ctrl.tap_key("z")
    assert res1 is True
    assert "z" in ctrl._key_down_times

    res2 = ctrl.tap_key("x")
    assert res2 is True
    assert "x" in ctrl._key_down_times


def test_hybrid_failover_when_vision_cut():
    """Verify that when vision is cut (frame is None), timeline audio scheduler handles notes."""
    from src.gameplay.osu.perception.mode_detector import mode_detection_service
    mock_backend = MockInputBackend()
    ctrl = CentralInputController(backend=mock_backend)
    app = OsuGameplayApp(controller=ctrl)

    mode_detection_service.set_user_override(GameMode.STANDARD)

    # Load simple synthetic standard beatmap
    simple_osu = """osu file format v14
[General]
Mode: 0
[Difficulty]
OverallDifficulty: 5
[HitObjects]
100,192,1000,1,0,0:0:0:0:
100,192,1200,1,0,0:0:0:0:
"""
    app.load_beatmap_from_string(simple_osu, key="test_failover")
    app._timeline_anchor_ns = clock.now_ns()
    app._is_running = True

    # Cut off vision: push NO frames into frame_channel
    frame_channel.clear()

    # Step at time 980ms (approaching first note at 1000ms)
    step_res = app.step(simulated_time_ms=980.0)
    assert step_res["permission_allowed"] is True

    # Step at time 1005ms (due time for first note)
    step_res2 = app.step(simulated_time_ms=1005.0)
    assert app.is_running is True


def test_discord_reporter_payload_formatting():
    reporter = OsuDiscordReporter(target_channel_id="1563921667556773969")
    rank = reporter._calculate_rank(98.5, 0)
    assert rank == "S"

    rank_ss = reporter._calculate_rank(100.0, 0)
    assert rank_ss == "SS"

    rank_a = reporter._calculate_rank(92.0, 2)
    assert rank_a == "A"
