"""
Unit Tests for Perception & Screen Capture Engine.
Tests Capture Backends, Playfield Coordinate Projection, and CaptureService.
"""

import time
import numpy as np
import pytest

from src.gameplay.osu.core.data_channels import frame_channel
from src.gameplay.osu.perception import (
    CaptureService,
    MockCaptureBackend,
    PlayfieldPreprocessor,
    ScreenFrame,
)


def test_mock_capture_backend():
    backend = MockCaptureBackend(width=800, height=600, color=(10, 20, 30))
    assert backend.is_active is False

    assert backend.start() is True
    assert backend.is_active is True

    frame = backend.grab()
    assert frame is not None
    assert frame.shape == (600, 800, 3)
    assert frame[0, 0, 0] == 10

    backend.stop()
    assert backend.is_active is False
    assert backend.grab() is None


def test_playfield_preprocessor_coordinates():
    prep = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    bounds = prep.bounds

    # Standard osu coordinate (0, 0) should map to top-left of playfield
    screen_x, screen_y = prep.osu_to_screen(0, 0)
    assert screen_x == bounds.x
    assert screen_y == bounds.y

    # Center coordinate (256, 192)
    center_sx, center_sy = prep.osu_to_screen(256, 192)
    assert center_sx > screen_x
    assert center_sy > screen_y

    # Roundtrip conversion
    osu_x, osu_y = prep.screen_to_osu(center_sx, center_sy)
    assert pytest.approx(osu_x, abs=1.0) == 256.0
    assert pytest.approx(osu_y, abs=1.0) == 192.0


def test_hit_circle_radius_calculation():
    prep = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)
    # Circle Size 4.0: osu radius = 54.4 - 4.48 * 4.0 = 36.48
    radius_cs4 = prep.get_hit_circle_radius(cs=4.0)
    # Circle Size 7.0 (smaller circles): radius = 54.4 - 4.48 * 7.0 = 23.04
    radius_cs7 = prep.get_hit_circle_radius(cs=7.0)

    assert radius_cs4 > radius_cs7
    assert radius_cs4 > 0.0


def test_crop_playfield_and_roi():
    prep = PlayfieldPreprocessor(screen_width=640, screen_height=480, playfield_height_ratio=0.8)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cropped = prep.crop_playfield(dummy_frame)
    assert cropped.shape[0] == prep.bounds.height
    assert cropped.shape[1] == prep.bounds.width

    roi_cropped = PlayfieldPreprocessor.crop_roi(dummy_frame, (10, 10, 50, 40))
    assert roi_cropped.shape == (40, 50, 3)


def test_capture_service_lifecycle():
    frame_channel.clear()
    service = CaptureService(backend_type="mock", fps=60)

    assert service.is_running is False
    assert service.start() is True
    assert service.is_running is True

    # Allow capture loop to run for a few frames
    time.sleep(0.08)

    service.stop()
    assert service.is_running is False
    assert service.frames_captured > 0

    latest_frame = frame_channel.get_latest()
    assert latest_frame is not None
    assert isinstance(latest_frame, ScreenFrame)
    assert latest_frame.width == 1280
    assert latest_frame.height == 720
    assert latest_frame.timestamp_ns > 0


def test_capture_service_single_frame():
    service = CaptureService(backend_type="mock", fps=30)
    frame = service.grab_single_frame()

    assert frame is not None
    assert isinstance(frame, ScreenFrame)
    assert frame.data.shape == (720, 1280, 3)


def test_capture_service_update_region():
    service = CaptureService(backend_type="mock", fps=30)
    service.start()
    try:
        # Dynamically update region to 800x600 at (100, 50)
        service.update_region((100, 50, 900, 650))
        time.sleep(0.05)
        frame = service.grab_single_frame()
        assert frame is not None
        assert frame.width == 800
        assert frame.height == 600
        assert frame.data.shape == (600, 800, 3)
    finally:
        service.stop()
