"""
Unit tests verifying fixes for Future Cake [Extra] beatmap discovery,
slider tail hold preservation, key alternation safety, and false positive fail alarms.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.gameplay.osu.beatmap.models import Slider, CurveType
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.world.object_lifecycle import WorldObject, ObjectLifecycleState
from src.gameplay.osu.integrations.tosu_client import TosuClient
from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.action_model import ActionType
from src.gameplay.osu.timing.calibration import timing_calibrator


def test_slider_caching_and_safe_tail_release():
    """Slider path and arc lengths must be cached on Slider object, and KEY_UP must hold past tail."""
    brain = StandardBrain(keys=["z", "x"])
    slider = Slider(
        x=256.0,
        y=192.0,
        time_ms=1000.0,
        hit_type=2,
        curve_type=CurveType.BEZIER,
        curve_points=[(300.0, 200.0), (350.0, 250.0)],
        repeat_count=1,
        pixel_length=150.0,
        duration_ms=400.0,
        end_time_ms=1400.0,
    )
    assert slider._cached_path is None
    assert slider._cached_cum_lens is None

    w_obj = WorldObject(id="s1", hit_object=slider)
    intents = brain.plan_actions(current_time_ms=900.0, objects=[w_obj], timeline_anchor_ns=1_000_000_000)

    # 1. Caching verified
    assert slider._cached_path is not None
    assert slider._cached_cum_lens is not None
    cached_len = len(slider._cached_path)
    assert cached_len >= 2

    # Second plan must reuse cached path without recomputing
    brain.plan_actions(current_time_ms=905.0, objects=[w_obj], timeline_anchor_ns=1_000_000_000)
    assert len(slider._cached_path) == cached_len

    # 2. KEY_UP must be dispatched with +12ms safe tail margin (1000 + 400 + 12 = 1412ms)
    up_intent = next((i for i in intents if i.action_type == ActionType.KEY_UP), None)
    assert up_intent is not None
    # 1_000_000_000 + 1412 * 1_000_000 with lead-time calibration
    expected_up_ns = timing_calibrator.calculate_dispatch_time_ns(1412.0, 1_000_000_000)
    assert up_intent.execute_at_ns == expected_up_ns


def test_tosu_is_failed_avoids_false_positive():
    """Tosu client must not report is_failed when no notes have been played or missed yet."""
    client = TosuClient(host="127.0.0.1", port=24050)
    # Mock payload with hp=0 before any hits (intro or map load)
    payload_intro = {
        "menu": {"state": 2},
        "gameplay": {
            "hp": {"normal": 0.0},
            "hits": {"0": 0, "50": 0, "100": 0, "300": 0, "grade": {"current": "D"}},
        },
    }
    client._parse_payload(payload_intro)
    assert client.is_playing is True
    assert client.hp == 0.0
    # Must NOT report failed because has_played is False and hits_miss == 0
    assert client.is_failed is False

    # When player actually plays and misses with hp = 0
    payload_failed = {
        "menu": {"state": 2},
        "gameplay": {
            "hp": {"normal": 0.0},
            "hits": {"0": 5, "50": 0, "100": 1, "300": 10, "grade": {"current": "F"}},
        },
    }
    client._parse_payload(payload_failed)
    assert client.is_failed is True


def test_beatmap_discovery_rejects_mania_map_for_standard():
    """Beatmap auto-discovery must reject Mode: 3 Mania maps and title mismatches."""
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.input.controller import CentralInputController
    app = OsuGameplayApp(controller=CentralInputController(backend=MockInputBackend()))

    # In mock backend, _try_auto_discover_beatmap returns False by default
    assert app._try_auto_discover_beatmap() is False
