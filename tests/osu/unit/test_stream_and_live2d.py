"""
Unit Tests for Stream Note Handling & Live2D Expression Activation in osu! Standard
"""

import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.data_channels import frame_channel
from src.gameplay.osu.core.state_store import GameMode, ModeDetectionStatus, ModeState, state_store
from src.gameplay.osu.perception.preprocessor import PlayfieldBounds
from src.gameplay.osu.perception.detector import VisionDetector
from tests.osu.unit.test_capture_and_key_timing import MockInputBackend


def test_standard_vision_detector_hit_circles():
    """Verify standard vision detector finds hit circles on playfield."""
    bounds = PlayfieldBounds(0, 0, 1280, 720, 1.0)
    detector = VisionDetector(bounds)
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Two standard hit circles
    cv2.circle(frame, (350, 250), 30, (240, 20, 20), -1)
    cv2.circle(frame, (500, 250), 30, (20, 120, 240), -1)

    t_ns = clock.now_ns()
    detected = detector.detect_objects(frame, timestamp_ns=t_ns)

    assert len(detected) == 2
    assert all(d.obj_type == "circle" for d in detected)


def test_optical_combo_milestone_triggers_live2d_expression():
    """Verify gameplay updates combo and triggers Live2D happy/excited emotion at milestone 10."""
    frame_channel.clear()
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    mock_input = MockInputBackend()
    app.controller.set_backend(mock_input)

    captured_emotions = []

    def mock_emit(emotion="neutral", intensity=1.0, duration=3.0, text="", reason=""):
        captured_emotions.append((emotion, reason, text))

    app._emit_emotion_and_thought = mock_emit
    app.start(session_id="test_combo_milestone", run_background_thread=False)

    try:
        # Simulate reaching combo milestone 10 directly from in-game data
        app.feedback.metrics.current_combo = 10
        app.feedback.metrics.count_300 = 10
        app.step()

        assert app.feedback.metrics.current_combo == 10
        assert app.feedback.metrics.accuracy == 100.0

        # Check that milestone 10 was reached and triggered happy emotion
        milestone_events = [e for e in captured_emotions if e[1] == "combo_milestone"]
        assert len(milestone_events) >= 1
        assert milestone_events[0][0] == "happy"
        assert "10 combo" in milestone_events[0][2]
    finally:
        app.stop()


def test_optical_combo_break_triggers_surprised_emotion():
    """Verify combo break after combo >= 15 triggers surprised emotion."""
    frame_channel.clear()
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    mock_input = MockInputBackend()
    app.controller.set_backend(mock_input)

    captured_emotions = []

    def mock_emit(emotion="neutral", intensity=1.0, duration=3.0, text="", reason=""):
        captured_emotions.append((emotion, reason, text))

    app._emit_emotion_and_thought = mock_emit
    app.start(session_id="test_combo_break", run_background_thread=False)

    try:
        # Pre-set combo to 20
        app.feedback.metrics.current_combo = 20
        app._last_reported_combo = 20

        # Simulate real in-game combo break: combo drops to 0
        app.feedback.metrics.current_combo = 0
        app.step()

        # Combo is 0
        assert app.feedback.metrics.current_combo == 0

        # Check combo break emotion
        break_events = [e for e in captured_emotions if e[1] == "combo_break"]
        assert len(break_events) >= 1
        assert break_events[0][0] == "surprised"
    finally:
        app.stop()
