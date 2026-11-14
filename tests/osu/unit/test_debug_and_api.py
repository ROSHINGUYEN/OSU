"""
Unit tests for osu! Visual Debug Logger, Tosu Integration, REST API, and Ground-Truth Override.
"""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.gameplay.osu.api.server import app
from src.gameplay.osu.beatmap.models import HitCircle
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.evaluation.debug_logger import VisualDebugLogger
from src.gameplay.osu.integrations.tosu_client import TosuClient
from src.gameplay.osu.perception.detector import DetectedObject
from src.gameplay.osu.perception.mode_detector import mode_detection_service
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.perception.tracker import TrackedObject
from src.gameplay.osu.world.reconciliation import ReconciliationService


def test_visual_debug_logger_annotation():
    """Test generating annotated debug frame and PNG encoding."""
    logger = VisualDebugLogger(output_dir="data/osu_debug_test")
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    det_katsu = DetectedObject(
        x=200.0, y=100.0, radius=20.0, confidence=0.9, timestamp_ns=1000,
        is_don=False, is_katsu=True, ring_r=10.0, ring_b=120.0,
    )
    det_don = DetectedObject(
        x=300.0, y=100.0, radius=20.0, confidence=0.9, timestamp_ns=1000,
        is_don=True, is_katsu=False, ring_r=140.0, ring_b=20.0,
    )

    annotated = logger.annotate_and_record(
        frame_rgb=frame,
        detected_objects=[det_katsu, det_don],
        mode="taiko",
        drum_line_x=112.0,
        accuracy=98.5,
        combo=35,
        offset_ms=-2.1,
        recent_action="D (act_test)",
    )

    assert annotated is not None
    assert annotated.shape == (300, 400, 3)

    png_bytes = logger.get_latest_png_bytes()
    assert png_bytes is not None
    assert len(png_bytes) > 100
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_standard_hit_circle_reconciliation():
    """Verify that visual detector and beatmap hit circle are cleanly reconciled in Standard mode."""
    reconciler = ReconciliationService()
    preprocessor = PlayfieldPreprocessor(800, 600)

    track = TrackedObject(
        track_id="trk_1",
        current_x=224.0,
        current_y=200.0,
        radius=25.0,
        approach_scale=1.0,
        obj_type="hit_circle",
        confidence=0.95,
        first_seen_ns=1000,
        last_seen_ns=2000,
    )

    bm_circle = HitCircle(
        x=256.0, y=192.0, time_ms=1500.0, hit_type=1,
        hit_sound=0,
    )

    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
    timeline = HitObjectTimeline([bm_circle])

    reconciled = reconciler.reconcile(
        current_time_ms=1500.0,
        tracked_objects=[track],
        preprocessor=preprocessor,
        timeline=timeline,
    )

    assert len(reconciled) == 1
    w_obj = reconciled[0]
    assert isinstance(w_obj.hit_object, HitCircle)
    assert w_obj.hit_object.time_ms == 1500.0


def test_tosu_client_parsing():
    """Test parsing standard Gosumemory/Tosu JSON payload."""
    client = TosuClient()
    mock_payload = {
        "menu": {
            "state": 2,
            "bm": {
                "path": {"file": "E:/osu/Songs/test.osu"}
            }
        },
        "gameplay": {
            "time": 24500.5,
            "combo": {"current": 42},
            "accuracy": 99.12,
            "hits": {"300": 80, "100": 3, "50": 0, "0": 1}
        }
    }

    client._parse_payload(mock_payload)
    snapshot = client.get_snapshot()

    assert snapshot["is_playing"] is True
    assert snapshot["time_ms"] == 24500.5
    assert snapshot["combo"] == 42
    assert snapshot["accuracy"] == 99.12
    assert snapshot["map_file_path"] == "E:/osu/Songs/test.osu"
    assert snapshot["hits"]["300"] == 80
    assert snapshot["hits"]["miss"] == 1


def test_osu_api_endpoints():
    """Test the FastAPI REST endpoints."""
    client = TestClient(app)

    # 1. Status
    r_status = client.get("/api/osu/status")
    assert r_status.status_code == 200
    assert "operating_state" in r_status.json()
    assert "tosu_snapshot" in r_status.json()

    # 2. Telemetry
    r_telem = client.get("/api/osu/telemetry")
    assert r_telem.status_code == 200
    assert "metrics" in r_telem.json()

    # 3. Command
    r_cmd = client.post("/api/osu/command", json={"command": "calibrate", "args": "3.5"})
    assert r_cmd.status_code == 200
    assert r_cmd.json()["success"] is True

    # 4. Web View HTML
    r_view = client.get("/api/osu/view")
    assert r_view.status_code == 200
    assert "text/html" in r_view.headers.get("content-type", "")
    assert "pov-image" in r_view.text
    assert "SHIORI AI — ĐỒNG HÀNH OSU! ZERO-LATENCY" in r_view.text


def test_shiori_vision_gui_retired():
    """Verify that obsolete Tkinter vision monitor GUI was cleanly removed in favor of 0ms memory gameplay."""
    from pathlib import Path
    gui_file = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "shiori_vision_gui.py"
    assert not gui_file.exists()
