import time
import numpy as np
import cv2
import pytest
from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.perception.capture import MockCaptureBackend
from src.gameplay.osu.perception.detector import VisionDetector
from src.gameplay.osu.perception.preprocessor import PlayfieldBounds


def test_osu_gameplay_app_background_thread_lifecycle():
    app = OsuGameplayApp()
    # Use mock backends for test isolation
    app.controller.set_backend(MockInputBackend())

    # Start session
    started = app.start(session_id="test_thread_loop")
    assert started is True
    assert app.is_running is True
    assert app._session_thread is not None
    assert app._session_thread.is_alive()

    # Wait a few frames
    time.sleep(0.05)

    # Stop session
    app.stop()
    assert app.is_running is False
    assert app._session_thread is None


def test_osu_live2d_emotions_and_thought_logging():
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())

    # 1. On Start
    app.start(session_id="test_emotions", run_background_thread=False)
    assert app._current_emotion == "excited"
    assert "Dạ Ba Roshi" in app._last_thought

    # 2. Simulate Combo Milestone 10
    app.feedback.metrics.current_combo = 10
    app.feedback.metrics.count_300 = 10
    step_out = app.step(simulated_time_ms=500.0)
    assert app._current_emotion == "happy"
    assert "10 combo" in app._last_thought
    assert step_out["current_emotion"] == "happy"

    # 3. Simulate Combo Milestone 100
    app.feedback.metrics.current_combo = 100
    app.step(simulated_time_ms=1000.0)
    assert app._current_emotion == "excited"
    assert "100x COMBO" in app._last_thought

    # 4. Simulate Combo Break
    app.feedback.metrics.current_combo = 0
    app.step(simulated_time_ms=1500.0)
    assert app._current_emotion == "surprised"
    assert "lỡ mất một nhịp" in app._last_thought

    # 5. On Stop
    app.stop()
    assert app._current_emotion == "happy"
    assert "vừa xong" in app._last_thought


def test_vision_detector_standard_hit_circle():
    bounds = PlayfieldBounds(x=0, y=0, width=1680, height=1050, scale=1.0)
    detector = VisionDetector(bounds)

    # Create synthetic frame with hit circle (Y = 400, X = 600)
    frame = np.zeros((1050, 1680, 3), dtype=np.uint8)
    cv2.circle(frame, (600, 400), 28, (255, 30, 30), -1)

    detected = detector.detect_objects(frame)
    assert len(detected) == 1
    assert abs(detected[0].x - 600.0) < 3.0
    assert abs(detected[0].y - 400.0) < 3.0
    assert detected[0].obj_type == "circle"


def test_machine_learning_adaptive_offset_learning_loop():
    """Verify adaptive machine learning feedback loop updates offset and metrics."""
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())
    app.feedback.calibrator.reset_offset()

    assert app.feedback.calibrator.adaptive_offset_ms == 0.0

    from src.gameplay.osu.core.state_store import GameMode
    # Start session
    app.start(session_id="test_ml_loop", mode_override=GameMode.STANDARD)
    try:
        from src.gameplay.osu.core.clock import clock
        from src.gameplay.osu.input.action_model import ActionIntent, ActionType

        real_now = clock.now_ns()
        target_time_ns = real_now - 10_000_000  # 10ms late
        act = ActionIntent(
            action_id="act_obj_ml_test_100",
            action_type=ActionType.KEY_TAP,
            execute_at_ns=target_time_ns,
            deadline_ns=real_now + 100_000_000,
            key="z",
        )
        app.queue.push(act)

        # Allow background session worker loop to pop and execute action
        for _ in range(30):
            if app.feedback.metrics.current_combo >= 1:
                break
            time.sleep(0.01)

        # Verify action was dispatched and machine learning calibrator updated
        assert app.feedback.metrics.current_combo >= 1
        # Error was ~10ms -> adaptive offset learned with alpha = 0.05
        assert app.feedback.calibrator.adaptive_offset_ms > 0.0
    finally:
        app.stop()


