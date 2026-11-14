"""
Perception Domain Package for Shiori AI osu! Gameplay.
Exports ScreenFrame, Capture Backends, CaptureService, and PlayfieldPreprocessor.
"""

from src.gameplay.osu.perception.capture import (
    BaseCaptureBackend,
    CaptureService,
    DXCamCaptureBackend,
    MSSCaptureBackend,
    MockCaptureBackend,
    ScreenFrame,
    capture_service,
)
from src.gameplay.osu.perception.detector import DetectedObject, VisionDetector
from src.gameplay.osu.perception.mode_detector import (
    ModeDetectionService,
    mode_detection_service,
)
from src.gameplay.osu.perception.preprocessor import (
    PlayfieldBounds,
    PlayfieldPreprocessor,
)
from src.gameplay.osu.perception.tracker import ObjectTracker, TrackedObject

__all__ = [
    "ScreenFrame",
    "BaseCaptureBackend",
    "DXCamCaptureBackend",
    "MSSCaptureBackend",
    "MockCaptureBackend",
    "CaptureService",
    "capture_service",
    "PlayfieldBounds",
    "PlayfieldPreprocessor",
    "DetectedObject",
    "VisionDetector",
    "TrackedObject",
    "ObjectTracker",
    "ModeDetectionService",
    "mode_detection_service",
]
