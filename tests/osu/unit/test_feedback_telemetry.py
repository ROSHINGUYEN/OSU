"""
Unit Tests for Phase 7: Gameplay Metrics, Feedback Evaluator, and Telemetry Logger.
"""

import json
import os
import shutil
import tempfile
import pytest

from src.gameplay.osu.beatmap.models import HitCircle
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.evaluation import (
    FeedbackEvaluator,
    GameplayMetrics,
    TelemetryLogger,
)
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject
from src.gameplay.osu.world.world_state import WorldStateManager


def test_gameplay_metrics_calculation():
    metrics = GameplayMetrics()

    # Hit 300, 300, 100, miss
    metrics.record_result("300", delta_ms=2.0)
    metrics.record_result("300", delta_ms=-1.5)
    metrics.record_result("100", delta_ms=12.0)
    metrics.record_result("miss", delta_ms=45.0)

    assert metrics.total_hits == 4
    assert metrics.count_300 == 2
    assert metrics.count_100 == 1
    assert metrics.count_miss == 1
    assert metrics.current_combo == 0  # Broken by miss
    assert metrics.max_combo == 3

    # Accuracy: (2*300 + 1*100) / (4*300) = 700 / 1200 = 58.33%
    assert pytest.approx(metrics.accuracy, abs=0.1) == 58.33

    # Unstable rate should be positive for varying timing deltas
    assert metrics.unstable_rate > 0.0


def test_gameplay_metrics_accuracy_setter():
    """Verify that accuracy can be set directly from external memory providers like Tosu."""
    metrics = GameplayMetrics()
    assert metrics.accuracy == 100.0

    metrics.accuracy = 97.45
    assert metrics.accuracy == 97.45

    metrics.reset()
    assert metrics.accuracy == 100.0


def test_feedback_evaluator_integration():
    world_mgr = WorldStateManager()
    evaluator = FeedbackEvaluator(world_mgr=world_mgr)

    hit = HitCircle(x=256, y=192, time_ms=1000, hit_type=1)
    w_obj = WorldObject("test_hit_1", hit, ObjectLifecycleState.ACTION_SENT)
    world_mgr.add_or_update(w_obj)

    # Hit with delta +10ms on OD=8 Standard -> Expect "300" (window is 32ms)
    res = evaluator.evaluate_hit("test_hit_1", delta_ms=10.0, od=8.0, mode=GameMode.STANDARD)
    assert res == "300"
    assert w_obj.lifecycle == ObjectLifecycleState.HIT_CONFIRMED
    assert w_obj.hit_result == "300"
    assert evaluator.metrics.current_combo == 1

    # Verify CentralStateStore received metrics update
    assert state_store.metrics.hit_counts["300"] >= 1


def test_telemetry_logger_jsonl():
    temp_dir = tempfile.mkdtemp()
    try:
        logger = TelemetryLogger(output_dir=temp_dir)
        logger.start_session("unit_test_session")

        logger.log_event("hit_confirmed", {"object_id": "obj_01", "score": 300})
        logger.log_event("combo_milestone", {"combo": 100})
        logger.stop_session()

        session_file = os.path.join(temp_dir, "session_unit_test_session.jsonl")
        assert os.path.isfile(session_file)

        # Read back
        replay = logger.load_replay(session_file)
        assert len(replay) >= 3  # start, 2 events, end
        assert replay[1]["event_type"] == "hit_confirmed"
        assert replay[1]["payload"]["score"] == 300
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
