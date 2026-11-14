"""
Unit Tests for Phase 4: Timing Engine, Calibration, Input Queue, Controller, and Emergency Stop.
"""

import time
import pytest

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, OperatingState, state_store
from src.gameplay.osu.input import (
    ActionIntent,
    ActionType,
    CentralInputController,
    CentralInputQueue,
    EmergencyStopManager,
    MockInputBackend,
)
from src.gameplay.osu.timing import (
    AdaptiveTimingCalibrator,
    DeadlineScheduler,
    HitWindowModel,
)


# ==========================================
# 1. Timing Tests
# ==========================================

def test_hit_window_model():
    # Standard OD=8.0
    w300, w100, w50 = HitWindowModel.standard(8.0)
    assert w300 == 80.0 - (6.0 * 8.0)  # 32.0ms
    assert w100 == 140.0 - (8.0 * 8.0)  # 76.0ms
    assert w50 == 200.0 - (10.0 * 8.0)  # 120.0ms

    # Evaluations
    assert HitWindowModel.evaluate_timing_error(15.0, 8.0, GameMode.STANDARD) == "300"
    assert HitWindowModel.evaluate_timing_error(50.0, 8.0, GameMode.STANDARD) == "100"
    assert HitWindowModel.evaluate_timing_error(100.0, 8.0, GameMode.STANDARD) == "50"
    assert HitWindowModel.evaluate_timing_error(150.0, 8.0, GameMode.STANDARD) == "miss"

    # Taiko OD=7.0 (ppy/osu DifficultyRange precision)
    w_great, w_good = HitWindowModel.taiko(7.0)
    assert w_great == 28.5
    assert w_good == 67.5
    assert HitWindowModel.evaluate_timing_error(20.0, 7.0, GameMode.TAIKO) == "300"
    assert HitWindowModel.evaluate_timing_error(40.0, 7.0, GameMode.TAIKO) == "100"


def test_adaptive_timing_calibrator():
    calibrator = AdaptiveTimingCalibrator(default_latency_ms=5.0, learning_rate=0.1, max_offset_ms=20.0)
    assert calibrator.adaptive_offset_ms == 0.0

    # If hits are consistently +10ms late, offset should increase to hit earlier
    calibrator.record_hit_sample(error_ms=10.0, confidence=0.9)
    assert calibrator.adaptive_offset_ms == pytest.approx(1.0, abs=0.01)

    # Extreme error clamping
    for _ in range(50):
        calibrator.record_hit_sample(error_ms=30.0, confidence=0.9)
    assert calibrator.adaptive_offset_ms == 20.0  # Clamped at max_offset_ms


def test_deadline_scheduler():
    t_start = clock.now_ns()
    target_ns = t_start + 4_000_000  # 4ms

    called = []
    def on_deadline():
        called.append(clock.now_ns())
        return "success"

    res, overshoot_ns = DeadlineScheduler.execute_at(target_ns, on_deadline)
    assert res == "success"
    assert len(called) == 1
    assert called[0] >= target_ns
    assert overshoot_ns >= 0


# ==========================================
# 2. Input Engine Tests
# ==========================================

def test_central_input_queue_priority_and_staleness():
    queue = CentralInputQueue(max_size=10)
    t_base = clock.now_ns()

    a1 = ActionIntent(
        action_id="act_1",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=t_base + 10_000_000,
        deadline_ns=t_base + 20_000_000,
        key="z",
    )
    a2 = ActionIntent(
        action_id="act_2",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=t_base + 5_000_000,   # Earlier deadline
        deadline_ns=t_base + 15_000_000,
        key="x",
    )

    queue.push(a1)
    queue.push(a2)
    assert len(queue) == 2

    # Earliest (a2) should be popped first when due
    assert queue.pop_due(t_base + 1_000_000) is None
    popped = queue.pop_due(t_base + 6_000_000)
    assert popped is not None
    assert popped.action_id == "act_2"

    # Stale action rejection
    popped_stale = queue.pop_due(t_base + 50_000_000)  # Passed deadline of a1 (20ms)
    assert popped_stale is None
    assert queue.stale_rejected_count == 1


def test_central_input_queue_enqueued_ids_lifecycle():
    """Verify that popping or pruning actions clears action_id from deduplication set."""
    queue = CentralInputQueue(max_size=10)
    t_base = clock.now_ns()

    a1 = ActionIntent(
        action_id="act_reuse",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=t_base,
        deadline_ns=t_base + 10_000_000,
        key="z",
    )
    assert queue.push(a1) is True
    # Duplicate push fails while in queue
    assert queue.push(a1) is False

    # Once popped, action_id is freed
    popped = queue.pop_due(t_base + 1000)
    assert popped is not None
    assert popped.action_id == "act_reuse"

    # Now pushing with the same action_id should succeed
    assert queue.push(a1) is True


def test_central_input_controller_and_stuck_key_watchdog():
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend, max_stuck_duration_ms=10.0)  # 10ms for fast test

    # Execute move
    move_act = ActionIntent(
        action_id="m1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=0,
        deadline_ns=1000,
        x=500.0,
        y=300.0,
    )
    assert controller.execute_action(move_act) is True
    assert backend.cursor_pos == (500, 300)

    # Press key "z"
    down_act = ActionIntent(
        action_id="k1",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=0,
        deadline_ns=1000,
        key="z",
    )
    assert controller.execute_action(down_act) is True
    assert "z" in backend.active_keys

    # Simulate key held past stuck duration (10ms)
    time.sleep(0.015)
    stuck_released = controller.check_stuck_keys()
    assert stuck_released == 1
    # Key should have been forcibly released by watchdog
    assert "z" not in backend.active_keys


def test_emergency_stop():
    backend = MockInputBackend()
    queue = CentralInputQueue()
    controller = CentralInputController(backend=backend)
    manager = EmergencyStopManager(controller=controller, queue=queue)

    # Fill queue and press keys
    queue.push(ActionIntent("act_pending", ActionType.KEY_DOWN, 100, 200, key="x"))
    controller.execute_action(ActionIntent("act_press", ActionType.KEY_DOWN, 0, 100, key="z"))
    assert len(queue) == 1
    assert "z" in backend.active_keys

    # Trigger emergency stop
    manager.trigger(reason="Test Stop")

    assert manager.is_triggered is True
    assert len(queue) == 0  # Queue wiped
    assert len(backend.active_keys) == 0  # Keys released
    assert state_store.system.state == OperatingState.SAFE_STOP
    assert state_store.safety.emergency_stop_triggered is True

    # Reset
    manager.reset()
    assert manager.is_triggered is False
    assert state_store.safety.emergency_stop_triggered is False
