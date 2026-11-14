"""
Unit Tests for Shiori AI osu! Comprehensive Learning System.
============================================================
Tests Feedback Provenance, 3-Tier Persistent Memory (Session, Beatmap Kalman, Skill),
12-Category Error Diagnostics Engine, Adaptive Planning Policy with Rollback,
and Contextual Strategy Learner.
"""

import os
import json
import pytest
from pathlib import Path

from src.gameplay.osu.evaluation.diagnostics import (
    DiagnosisReport,
    ErrorCategory,
    GameplayDiagnosticsEngine,
    diagnostics_engine,
)
from src.gameplay.osu.evaluation.feedback import (
    FeedbackEvaluator,
    ObservationSource,
)
from src.gameplay.osu.evaluation.game_memory import OsuGameMemory
from src.gameplay.osu.evaluation.metrics import GameplayMetrics
from src.gameplay.osu.evaluation.strategy_learner import (
    StrategyLearner,
    strategy_learner,
)
from src.gameplay.osu.planning.adaptive_policy import (
    AdaptivePlanningPolicy,
    PolicyBounds,
    adaptive_policy,
)
from src.gameplay.osu.core.state_store import GameMode


def test_feedback_provenance_and_deduplication():
    """Verify ObservationSource tagging and deduplication of cumulative Tosu hit errors."""
    fb = FeedbackEvaluator()
    assert fb.provenance == ObservationSource.UNKNOWN

    # 1. First packet of hit errors from Tosu
    packet1 = [-4.0, 2.0, 5.0]
    fb.record_tosu_hit_errors(packet1, source=ObservationSource.OBSERVED)
    assert fb.provenance == ObservationSource.OBSERVED
    assert len(fb.metrics.timing_errors_ms) == 3
    assert len(fb.verified_timing_errors_ms) == 3
    assert abs(fb.metrics.timing_error_mean - 1.0) < 1e-4

    # 2. Second cumulative packet with 2 new errors (5 total)
    packet2 = [-4.0, 2.0, 5.0, 3.0, -1.0]
    fb.record_tosu_hit_errors(packet2, source=ObservationSource.OBSERVED)
    # Deduplication must ensure only 2 new errors were appended
    assert len(fb.metrics.timing_errors_ms) == 5
    assert len(fb.verified_timing_errors_ms) == 5
    assert fb.metrics.timing_error_variance > 0.0
    assert fb.metrics.unstable_rate > 0.0

    # 3. Reset
    fb.reset()
    assert fb.provenance == ObservationSource.UNKNOWN
    assert len(fb.metrics.timing_errors_ms) == 0
    assert len(fb.verified_timing_errors_ms) == 0


def test_kalman_filter_bayesian_math_and_direction(tmp_path):
    """Verify 1D Bayesian Kalman filter updates direction, reduces uncertainty, and enforces bounds."""
    test_json = tmp_path / "test_profiles.json"
    memory = OsuGameMemory()
    memory.__class__.profile_file = property(lambda self: test_json)

    session1 = {
        "beatmap_id": 12345,
        "beatmap_title": "Freedom Dive",
        "mode": "standard",
        "accuracy": 95.0,
        "combo": 500,
        "miss_count": 2,
        "duration_s": 120.0,
        "avg_timing_error_ms": 5.0,  # Late hits (+5ms)
        "timing_error_variance": 4.0,
        "timing_errors": [5.0] * 50,
        "is_stream_heavy": True,
    }

    prof1 = memory.update_beatmap_profile_kalman(session1)
    assert prof1["total_plays"] == 1
    # Late error (+5.0ms) must increase lead offset (>0) so notes are hit earlier
    assert prof1["learned_adaptive_offset_ms"] > 0.0
    assert prof1["archetype"] == "stream_speed"
    var1 = prof1["kalman_variance"]

    # Second session with same beatmap
    session2 = {
        "beatmap_id": 12345,
        "beatmap_title": "Freedom Dive",
        "mode": "standard",
        "accuracy": 98.0,
        "combo": 1200,
        "miss_count": 0,
        "duration_s": 120.0,
        "avg_timing_error_ms": 1.0,
        "timing_error_variance": 1.5,
        "timing_errors": [1.0] * 80,
    }
    prof2 = memory.update_beatmap_profile_kalman(session2)
    assert prof2["total_plays"] == 2
    # Additional reliable observation must reduce estimation uncertainty
    assert prof2["kalman_variance"] < var1
    assert prof2["best_accuracy"] == 98.0
    # Clamped bounds check
    assert -6.0 <= prof2["learned_adaptive_offset_ms"] <= 6.0


def test_kalman_filter_quality_gates_reject_quits_and_noise(tmp_path):
    """Verify incomplete plays (<5 hits or <2.0s) do not corrupt the Kalman filter offset."""
    test_json = tmp_path / "test_profiles_gate.json"
    memory = OsuGameMemory()
    memory.__class__.profile_file = property(lambda self: test_json)

    # 1. Establish valid baseline
    memory.update_beatmap_profile_kalman({
        "beatmap_id": 888,
        "beatmap_title": "Test Map",
        "mode": "standard",
        "accuracy": 96.0,
        "combo": 200,
        "duration_s": 45.0,
        "avg_timing_error_ms": 3.0,
        "hits": {"300": 200},
    })
    baseline_offset = memory.get_beatmap_profile(888)["learned_adaptive_offset_ms"]
    assert baseline_offset > 0.0

    # 2. Player quits after 2 hits / 1.5 seconds with extreme noise (+25ms)
    abort_session = {
        "beatmap_id": 888,
        "beatmap_title": "Test Map",
        "mode": "standard",
        "accuracy": 50.0,
        "combo": 2,
        "miss_count": 1,
        "duration_s": 1.5,
        "avg_timing_error_ms": 25.0,
        "result": "quit",
        "hits": {"300": 2, "miss": 1},
    }
    prof_after_abort = memory.update_beatmap_profile_kalman(abort_session)

    # Offset must be UNCHANGED by the abort session
    assert prof_after_abort["learned_adaptive_offset_ms"] == baseline_offset
    assert prof_after_abort["total_quits"] == 1
    assert prof_after_abort["total_plays"] == 1  # Not counted as full play


def test_3_tier_memory_mode_isolation_and_legacy_migration(tmp_path):
    """Verify beatmap profiles are isolated across game modes and migrate legacy keys."""
    test_json = tmp_path / "test_profiles_modes.json"
    memory = OsuGameMemory()
    memory.__class__.profile_file = property(lambda self: test_json)

    # Same beatmap ID 777 played in Standard and Taiko
    memory.update_beatmap_profile_kalman({
        "beatmap_id": 777,
        "beatmap_title": "Shared Map",
        "mode": "standard",
        "accuracy": 92.0,
        "combo": 300,
        "duration_s": 60.0,
        "avg_timing_error_ms": 4.0,
    })

    memory.update_beatmap_profile_kalman({
        "beatmap_id": 777,
        "beatmap_title": "Shared Map",
        "mode": "taiko",
        "accuracy": 99.0,
        "combo": 500,
        "duration_s": 60.0,
        "avg_timing_error_ms": -2.0,
    })

    prof_std = memory.get_beatmap_profile(777, mode="standard")
    prof_taiko = memory.get_beatmap_profile(777, mode="taiko")

    assert prof_std is not None
    assert prof_taiko is not None
    # Profiles must NOT overwrite each other
    assert prof_std["mode"] == "standard"
    assert prof_taiko["mode"] == "taiko"
    assert prof_std["learned_adaptive_offset_ms"] > 0.0
    assert prof_taiko["learned_adaptive_offset_ms"] < 0.0
    assert prof_taiko["archetype"] == "rhythm_percussion"


def test_level_c_skill_memory_cross_beatmap_learning(tmp_path):
    """Verify Level C Skill Memory tracks mechanic proficiencies across beatmaps."""
    test_skill_json = tmp_path / "test_skills.json"
    memory = OsuGameMemory()
    memory.__class__.skill_file = property(lambda self: test_skill_json)

    # Record 1: stream_speed with 90% acc, +4ms error
    skill1 = memory.record_skill_performance("stream_speed", {
        "accuracy": 90.0,
        "avg_timing_error_ms": 4.0,
    })
    assert skill1["samples_count"] == 1
    assert skill1["avg_accuracy"] == 90.0
    assert skill1["proficiency_score"] > 0.6

    # Record 2: improved stream performance
    skill2 = memory.record_skill_performance("stream_speed", {
        "accuracy": 98.0,
        "avg_timing_error_ms": 1.0,
    })
    assert skill2["samples_count"] == 2
    assert skill2["avg_accuracy"] > 90.0
    assert skill2["proficiency_score"] > skill1["proficiency_score"]

    # Verify query
    loaded = memory.get_skill_profile("stream_speed")
    assert loaded is not None
    assert loaded["samples_count"] == 2


def test_error_diagnostics_taxonomy_classification():
    """Verify GameplayDiagnosticsEngine correctly categorizes the 12 failure categories."""
    engine = GameplayDiagnosticsEngine()

    # 1. TIMING_LATE
    d_late = engine.diagnose_session({
        "accuracy": 78.0,
        "miss_count": 4,
        "avg_timing_error_ms": 16.5,
        "mode": "standard",
    })
    assert d_late.primary_category == ErrorCategory.TIMING_LATE
    assert "TRỄ" in d_late.evidence[0]

    # 2. TIMING_EARLY
    d_early = engine.diagnose_session({
        "accuracy": 82.0,
        "miss_count": 3,
        "avg_timing_error_ms": -14.0,
        "mode": "standard",
    })
    assert d_early.primary_category == ErrorCategory.TIMING_EARLY
    assert "SỚM" in d_early.evidence[0]

    # 3. AIM_ERROR
    d_aim = engine.diagnose_session({
        "accuracy": 88.0,
        "miss_count": 5,
        "avg_timing_error_ms": 1.2,  # Timing is almost perfect
        "mode": "standard",
    })
    assert d_aim.primary_category == ErrorCategory.AIM_ERROR

    # 4. SLIDER_TRACKING_ERROR
    d_slider = engine.diagnose_session({
        "accuracy": 85.0,
        "miss_count": 3,
        "mode": "standard",
        "hits": {"sliderBreaks": 3},
    })
    assert d_slider.primary_category == ErrorCategory.SLIDER_TRACKING_ERROR

    # 5. ACTION_DEADLINE_MISS / QUEUE_OVERLOAD
    d_queue = engine.diagnose_session({
        "accuracy": 60.0,
        "miss_count": 10,
        "tuning_applied": {"stale_actions_rejected": 8},
        "mode": "standard",
    })
    assert d_queue.primary_category in (ErrorCategory.QUEUE_OVERLOAD, ErrorCategory.ACTION_DEADLINE_MISS)

    # 6. TELEMETRY_UNAVAILABLE
    d_telem = engine.diagnose_session({
        "accuracy": 0.0,
        "duration_s": 10.0,
        "provenance": "unknown",
        "failure_reason": "telemetry_lost",
    })
    assert d_telem.primary_category == ErrorCategory.TELEMETRY_UNAVAILABLE


def test_adaptive_planning_policy_and_regression_rollback():
    """Verify AdaptivePlanningPolicy bounds, parameter updates, and automatic rollback."""
    policy = AdaptivePlanningPolicy()
    policy.reset()

    # 1. Update from TIMING_LATE -> should increment lead_time_offset_ms
    diag_late = DiagnosisReport(primary_category=ErrorCategory.TIMING_LATE)
    adj = policy.update_from_diagnosis(diag_late)
    assert adj["lead_time_offset_ms"] == 0.5
    assert policy.get_param("lead_time_offset_ms") == 0.5

    # 2. Update from AIM_ERROR -> should increment aim_smoothing_steps and preparation_lead_ms
    diag_aim = DiagnosisReport(primary_category=ErrorCategory.AIM_ERROR)
    adj_aim = policy.update_from_diagnosis(diag_aim)
    assert adj_aim["aim_smoothing_steps"] == 4
    assert adj_aim["preparation_lead_ms"] == 9.0

    # 3. Regression test: Accuracy dropped sharply from 95% to 80% (drop = 15% > threshold 6%)
    rolled_back = policy.check_and_rollback_regression(current_accuracy=80.0, previous_accuracy=95.0)
    assert rolled_back is True
    # Parameters must revert to baseline before the bad adjustment
    assert policy.get_param("aim_smoothing_steps") == 3

    # Reset
    policy.reset()
    assert policy.get_param("lead_time_offset_ms") == 0.0


def test_strategy_learner_contextual_recommendations_and_offline_replay():
    """Verify StrategyLearner produces archetype strategies and accurate replay evaluation."""
    learner = StrategyLearner()

    # 1. Archetype strategy recommendations
    strat_stream = learner.recommend_strategy("stream_speed")
    assert strat_stream["aim_smoothing_steps"] == 4
    assert strat_stream["deadline_margin_ms"] == 30.0

    strat_taiko = learner.recommend_strategy("rhythm_percussion")
    assert strat_taiko["aim_smoothing_steps"] == 1

    # 2. Offline replay evaluation
    mock_session = {
        "timing_errors": [15.0, 20.0, 18.0, 12.0, 25.0],  # Consistently late ~18ms
        "avg_timing_error_ms": 18.0,
    }

    # Evaluate replay with +15ms candidate offset
    replay_res = learner.evaluate_replay_session(mock_session, candidate_offset_ms=15.0)
    # Mean error should be shifted from 18ms down to ~3ms
    assert abs(replay_res["simulated_mean_error_ms"] - 3.0) < 0.1
    # Accuracy must be substantially higher under candidate offset
    assert replay_res["simulated_accuracy"] > 80.0


def test_adaptive_policy_key_state_error_tuning():
    """Verify AdaptivePlanningPolicy tunes alternation and post-slider thresholds upon key state errors."""
    policy = AdaptivePlanningPolicy()
    policy.reset()

    assert policy.get_param("alternation_threshold_ms") == 200.0
    assert policy.get_param("post_slider_gap_ms") == 260.0

    diag_key = DiagnosisReport(primary_category=ErrorCategory.KEY_STATE_ERROR)
    adj = policy.update_from_diagnosis(diag_key)

    assert adj["alternation_threshold_ms"] == 215.0
    assert adj["post_slider_gap_ms"] == 280.0
    assert policy.get_param("alternation_threshold_ms") == 215.0
    assert policy.get_param("post_slider_gap_ms") == 280.0


def test_closed_loop_log_session_auto_diagnoses_and_fine_tunes(tmp_path):
    """Verify log_session automatically performs diagnostic categorization and fine-tunes policy."""
    test_jsonl = tmp_path / "test_memory.jsonl"
    memory = OsuGameMemory()
    memory.__class__.memory_file = property(lambda self: test_jsonl)

    # Session with late timing error and misses
    session_data = {
        "beatmap_id": 99999,
        "beatmap_title": "Test Closed Loop",
        "mode": "standard",
        "result": "fail",
        "accuracy": 75.0,
        "combo": 80,
        "max_combo": 100,
        "miss_count": 6,
        "avg_timing_error_ms": 15.0,
        "timing_error_variance": 20.0,
        "provenance": "observed",
        "duration_s": 45.0,
        "failure_reason": "hp_zero",
    }

    memory.log_session(session_data)

    # Read back log
    history = memory.get_history(beatmap_id=99999)
    assert len(history) == 1
    logged_entry = history[0]

    # Diagnostics must NOT be None!
    assert logged_entry["diagnostics"] is not None
    assert logged_entry["diagnostics"]["primary_category"] == "TIMING_LATE"
    assert "TRỄ" in logged_entry["diagnostics"]["evidence"][0]


def test_adaptive_policy_json_persistence(tmp_path):
    """Verify AdaptivePlanningPolicy persists parameters to disk and restores them."""
    test_json = tmp_path / "test_policy.json"
    
    # Instance 1: update parameters
    p1 = AdaptivePlanningPolicy()
    p1.params = dict(AdaptivePlanningPolicy.DEFAULT_PARAMS)
    p1.__class__.policy_file = property(lambda self: test_json)
    
    diag_aim = DiagnosisReport(primary_category=ErrorCategory.AIM_ERROR)
    p1.update_from_diagnosis(diag_aim)
    assert p1.get_param("aim_lead_time_ms") == 18.0
    assert p1.get_param("aim_smoothing_steps") == 4
    assert test_json.exists()

    # Instance 2: verify parameters restored from disk
    p2 = AdaptivePlanningPolicy()
    p2.__class__.policy_file = property(lambda self: test_json)
    p2.load_policy()
    assert p2.get_param("aim_lead_time_ms") == 18.0
    assert p2.get_param("aim_smoothing_steps") == 4


def test_standard_key_safety_guard_blocks_non_standard_keys():
    """Verify OsuGameplayApp blocks dispatching non-standard keys in Standard mode."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.input.action_model import ActionIntent, ActionType
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.core.state_store import state_store, GameMode
    from src.gameplay.osu.perception.mode_detector import mode_detection_service
    from src.gameplay.osu.core.clock import clock

    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app.queue.clear()

    mode_detection_service.evaluate_mode(beatmap_mode=GameMode.STANDARD, confidence=1.0)
    app.start(session_id="test_key_guard", run_background_thread=False)

    try:
        now = clock.now_ns()
        # Queue a valid 'z' action and an invalid 'd' (Taiko) action
        act_valid = ActionIntent("act_valid", ActionType.KEY_TAP, now - 1000, now + 100000000, key="z")
        act_invalid = ActionIntent("act_invalid", ActionType.KEY_TAP, now - 1000, now + 100000000, key="d")

        app.queue.push(act_valid)
        app.queue.push(act_invalid)

        # Step at time when both are due
        app.step(simulated_time_ms=100.0)

        # Backend history must ONLY contain 'z', and NEVER 'd'
        executed_keys = [entry[1] for entry in backend.history if entry[0] == "key_down"]
        assert "z" in executed_keys
        assert "d" not in executed_keys
    finally:
        app.stop()


