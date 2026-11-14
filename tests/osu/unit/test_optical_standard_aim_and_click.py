"""
Unit Tests for Direct Memory-Driven Standard Aim and Click Engine
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    state_store,
)
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.beatmap.models import BeatmapData, HitCircle
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline


def test_optical_standard_aims_at_lowest_approach_scale():
    """Verify Standard Engine aims at Target #1 and clicks when due."""
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app.auto_aim_enabled = True

    bm = BeatmapData(
        mode=GameMode.STANDARD,
        hit_objects=[
            HitCircle(x=200.0, y=150.0, time_ms=1000.0, hit_type=1),
            HitCircle(x=400.0, y=300.0, time_ms=2000.0, hit_type=1),
        ],
    )
    app._timeline = HitObjectTimeline(bm.hit_objects)

    app.start(session_id="test_std_aim", run_background_thread=False)
    app.preprocessor.update_window_position(100, 50)
    try:
        # Step at t=1000ms: Hits Target 1
        app.step(simulated_time_ms=1000.0)

        # Check cursor movement and key tap
        move_events = [h for h in backend.history if h[0] == "move"]
        key_events = [h for h in backend.history if h[0] == "key_down"]
        assert len(move_events) >= 1
        assert len(key_events) >= 1
        assert key_events[0][1] in ("z", "x")
    finally:
        app.stop()


def test_optical_standard_advances_to_next_target_after_hit():
    """Verify engine transitions to Target #2 once Target #1 is executed."""
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.STANDARD, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="system",
    )
    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app.auto_aim_enabled = True

    bm = BeatmapData(
        mode=GameMode.STANDARD,
        hit_objects=[
            HitCircle(x=200.0, y=150.0, time_ms=1000.0, hit_type=1),
            HitCircle(x=400.0, y=300.0, time_ms=1500.0, hit_type=1),
        ],
    )
    app._timeline = HitObjectTimeline(bm.hit_objects)

    app.start(session_id="test_std_seq", run_background_thread=False)
    app.preprocessor.update_window_position(100, 50)
    try:
        # Step 1: Hits Target 1 at 1000ms
        app.step(simulated_time_ms=1000.0)
        assert len([h for h in backend.history if h[0] == "key_down"]) >= 1

        # Step 2: Hits Target 2 at 1500ms
        app.step(simulated_time_ms=1500.0)
        assert len([h for h in backend.history if h[0] == "key_down"]) >= 2
    finally:
        app.stop()


def test_optical_standard_alternates_keys():
    """Verify rapid stream hits alternate between 'z' and 'x'."""
    from src.gameplay.osu.brains.standard_brain import StandardBrain
    from src.gameplay.osu.world.object_lifecycle import WorldObject

    brain = StandardBrain(keys=["z", "x"])
    c1 = HitCircle(x=200.0, y=200.0, time_ms=1000.0, hit_type=1)
    c2 = HitCircle(x=400.0, y=200.0, time_ms=1090.0, hit_type=1)  # 90ms (stream/burst <= 145ms)

    w1 = WorldObject(id="c1", hit_object=c1)
    w2 = WorldObject(id="c2", hit_object=c2)
    anchor = clock.now_ns()

    intents1 = brain.plan_actions(current_time_ms=900.0, objects=[w1], timeline_anchor_ns=anchor)
    intents2 = brain.plan_actions(current_time_ms=1050.0, objects=[w2], timeline_anchor_ns=anchor)

    key1 = [i.key for i in intents1 if i.key][0]
    key2 = [i.key for i in intents2 if i.key][0]

    assert {key1, key2} == {"z", "x"}
    assert key1 != key2
