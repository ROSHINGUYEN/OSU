"""
Unit Tests for Phase 6: Rolling-Horizon Planner, Utility Evaluator, Risk Model, and Action Validator.
"""

import pytest

from src.gameplay.osu.beatmap.models import HitCircle
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.input.action_model import ActionIntent, ActionType
from src.gameplay.osu.input.queue import CentralInputQueue
from src.gameplay.osu.planning import (
    ActionValidator,
    RiskModel,
    RollingHorizonPlanner,
    UtilityEvaluator,
)
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject
from src.gameplay.osu.world.world_state import WorldStateManager


def test_utility_evaluator():
    evaluator = UtilityEvaluator()
    dummy_act = ActionIntent("a1", ActionType.KEY_TAP, 0, 100, key="z")

    # High combo gives bonus
    score_low_combo = evaluator.evaluate(dummy_act, current_combo=5, prob_hit=0.9, risk_next=0.05)
    score_high_combo = evaluator.evaluate(dummy_act, current_combo=100, prob_hit=0.9, risk_next=0.05)

    assert score_high_combo > score_low_combo


def test_risk_model_unsalvageable_drop():
    # Normal distance 100px in 100ms -> 1.0 px/ms -> reachable
    reach_prob = RiskModel.evaluate_reachability(0, 0, 100, 0, time_to_target_ms=100.0)
    assert reach_prob > 0.95
    assert RiskModel.is_unsalvageable(0, 0, 100, 0, time_to_target_ms=100.0) is False

    # Extreme jump: 800px in 30ms -> 26.6 px/ms -> unreachable & impossible
    unsalvageable = RiskModel.is_unsalvageable(
        curr_x=0, curr_y=0, target_x=800, target_y=0,
        time_to_target_ms=30.0, dist_to_next=500.0, time_to_next_ms=50.0
    )
    assert unsalvageable is True


def test_action_validator():
    validator = ActionValidator(screen_width=1920, screen_height=1080)

    # Valid action
    valid_act = ActionIntent("v1", ActionType.MOUSE_MOVE, 100, 200, x=500, y=500)
    ok, reason = validator.validate(valid_act)
    assert ok is True
    assert reason is None

    # Invalid coordinates (out of bounds)
    oob_act = ActionIntent("oob", ActionType.MOUSE_MOVE, 100, 200, x=-50, y=500)
    ok, reason = validator.validate(oob_act)
    assert ok is False
    assert "out of bounds" in reason

    # Invalid deadline (execute_at > deadline)
    inverted_act = ActionIntent("inv", ActionType.KEY_DOWN, 250, 200, key="z")
    ok, reason = validator.validate(inverted_act)
    assert ok is False
    assert "Invalid timing" in reason


def test_rolling_horizon_planner_cycle():
    queue = CentralInputQueue()
    world_mgr = WorldStateManager()
    planner = RollingHorizonPlanner(
        horizon_ms=1000.0,
        commit_window_ms=200.0,
        queue=queue,
        world_manager=world_mgr,
    )

    anchor_ns = clock.now_ns()

    # Add 2 Standard hit circles to world manager
    n1 = HitCircle(x=256, y=192, time_ms=50.0, hit_type=1)   # In near-term commit window (50ms)
    n2 = HitCircle(x=256, y=192, time_ms=800.0, hit_type=1)  # Far horizon (800ms)

    w1 = WorldObject("c1", n1, ObjectLifecycleState.APPROACHING)
    w2 = WorldObject("c2", n2, ObjectLifecycleState.PREDICTED)

    world_mgr.add_or_update(w1)
    world_mgr.add_or_update(w2)

    # Plan cycle at current_time_ms = 0.0 in Standard mode
    committed = planner.plan_cycle(current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)

    # Note 1 should be committed to queue because it is within 200ms commit window
    assert len(committed) >= 1
    assert len(queue) >= 1
    # Check that note 1 lifecycle transitioned to ACTION_SCHEDULED
    assert world_mgr.get_object("c1").lifecycle == ObjectLifecycleState.ACTION_SCHEDULED

    # Verify CentralStateStore updated
    assert state_store.planner.planning_horizon_ms == 1000.0
    assert state_store.planner.replan_counter == 1


def test_rolling_horizon_planner_standard_mode_full_sequence_commit():
    """Verify that when a HitCircle or Slider enters commitment, ALL actions in its trajectory are committed."""
    from src.gameplay.osu.beatmap.models import HitCircle, Slider
    from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor

    queue = CentralInputQueue()
    world_mgr = WorldStateManager()
    prep = PlayfieldPreprocessor(1280, 720)
    planner = RollingHorizonPlanner(
        horizon_ms=1000.0,
        commit_window_ms=150.0,
        queue=queue,
        world_manager=world_mgr,
        preprocessor=prep,
    )

    anchor_ns = clock.now_ns()

    # Note 1: HitCircle at 160ms. Movement starts at ~100ms (<= 150ms commit window).
    c1 = HitCircle(x=300.0, y=200.0, time_ms=160.0, hit_type=1)
    w1 = WorldObject("c1", c1, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w1)

    committed = planner.plan_cycle(current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)

    # All actions for c1 (moves + tap + click) must be committed together!
    action_types = [a.action_type for a in committed]
    assert ActionType.KEY_TAP in action_types, "KEY_TAP was dropped by commitment truncation!"
    assert ActionType.MOUSE_MOVE in action_types, "MOUSE_MOVE was dropped by commitment truncation!"
    assert ActionType.MOUSE_CLICK not in action_types, "MOUSE_CLICK must not be emitted to preserve slider integrity!"
    move_actions = [a for a in committed if a.action_type == ActionType.MOUSE_MOVE]
    assert len(move_actions) >= 2, f"Expected full aiming move steps, got {len(move_actions)}"

    # Check world object transitioned to ACTION_SCHEDULED
    assert world_mgr.get_object("c1").lifecycle == ObjectLifecycleState.ACTION_SCHEDULED

    # On next cycle (10ms later), c1 must not generate duplicate actions and queue must retain all items
    committed_cycle_2 = planner.plan_cycle(current_time_ms=10.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)
    assert len(committed_cycle_2) == 0, "Expected 0 duplicate actions for already-scheduled object"
    assert len(queue) == len(committed), "Queue must retain all committed actions"


def test_rolling_horizon_planner_reset_allows_rescheduling():
    """Verify that calling planner.reset() allows objects to be re-scheduled on retry."""
    queue = CentralInputQueue()
    world_mgr = WorldStateManager()
    planner = RollingHorizonPlanner(
        horizon_ms=1000.0,
        commit_window_ms=150.0,
        queue=queue,
        world_manager=world_mgr,
    )
    anchor_ns = 1_000_000_000
    n1 = HitCircle(x=256, y=192, time_ms=100.0, hit_type=1)
    w1 = WorldObject("c1", n1, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w1)

    # First run
    committed1 = planner.plan_cycle(current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)
    assert len(committed1) >= 1
    assert len(queue) >= 1

    # Simulate retry without reset -> 0 committed because action_id was committed
    queue.clear()
    committed2 = planner.plan_cycle(current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)
    assert len(committed2) == 0

    # Reset planner and world object lifecycle -> can re-commit
    planner.reset()
    w1_retry = WorldObject("c1", n1, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w1_retry)
    committed3 = planner.plan_cycle(current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD)
    assert len(committed3) >= 1
    assert len(queue) >= 1


