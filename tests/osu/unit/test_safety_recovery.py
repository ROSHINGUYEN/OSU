"""
Unit Tests for Phase 8: Permission Leases, Safety Supervisor, and Recovery FSM/Orchestrator.
"""

import time
import pytest

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    OperatingState,
    RecoveryFsmState,
    state_store,
)
from src.gameplay.osu.input import (
    ActionIntent,
    ActionType,
    CentralInputController,
    CentralInputQueue,
    MockInputBackend,
)
from src.gameplay.osu.recovery import RecoveryFsm, RecoveryOrchestrator
from src.gameplay.osu.safety import (
    PermissionDecisionType,
    PermissionEngine,
    SafetySupervisor,
)


def test_permission_engine_leases_and_revocation():
    engine = PermissionEngine(default_lease_ms=20)  # 20ms TTL

    decision = engine.grant_lease()
    assert decision.is_allowed is True
    assert decision.lease_id != ""
    assert engine.validate_lease(decision.lease_id) is True

    # After TTL, lease expires
    time.sleep(0.025)
    assert engine.validate_lease(decision.lease_id) is False

    # Active revocation
    d2 = engine.grant_lease()
    assert engine.validate_lease(d2.lease_id) is True
    revoked_count = engine.revoke_all(reason="Test active revocation")
    assert revoked_count >= 1
    assert engine.validate_lease(d2.lease_id) is False


def test_safety_supervisor_dependency_evaluation_and_revocation():
    engine = PermissionEngine()
    queue = CentralInputQueue()
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend)
    supervisor = SafetySupervisor(engine=engine, queue=queue, controller=controller)

    # 1. When mode is CONFIRMED, request_action_permission grants lease
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.TAIKO, status=ModeDetectionStatus.MODE_CONFIRMED),
        caller="mode_detector",
    )
    decision = supervisor.request_action_permission()
    assert decision.decision == PermissionDecisionType.ALLOW

    # 2. When mode becomes UNKNOWN (degraded dependency)
    state_store.mutate_domain(
        "mode",
        lambda curr: ModeState(mode=GameMode.UNKNOWN, status=ModeDetectionStatus.UNKNOWN),
        caller="mode_detector",
    )
    # Add a pending action to queue
    queue.push(ActionIntent("act_pending", ActionType.KEY_DOWN, 100, 200, key="z"))
    assert len(queue) == 1

    # Request permission under degraded conditions
    denied = supervisor.request_action_permission()
    assert denied.decision == PermissionDecisionType.DENY
    assert "MODE_NOT_CONFIRMED" in denied.reason_codes

    # Active Revocation should have wiped queue and set state to DEGRADED
    assert len(queue) == 0
    assert state_store.system.state == OperatingState.DEGRADED


def test_recovery_fsm_lifecycle():
    fsm = RecoveryFsm(max_retries=2)
    assert fsm.state == RecoveryFsmState.IDLE

    # Request recovery
    assert fsm.request_recovery("stuck_key") is True
    assert fsm.state == RecoveryFsmState.REQUESTED

    # Start recovering
    fsm.start_recovering()
    assert fsm.state == RecoveryFsmState.RECOVERING

    # Attempt 1 fails -> BACKOFF
    st1 = fsm.record_attempt_failed()
    assert st1 == RecoveryFsmState.BACKOFF
    assert fsm.retry_count == 1

    # Attempt 2 fails -> BACKOFF
    fsm.start_recovering()
    st2 = fsm.record_attempt_failed()
    assert st2 == RecoveryFsmState.BACKOFF
    assert fsm.retry_count == 2

    # Attempt 3 fails (> max_retries 2) -> FAILED
    fsm.start_recovering()
    st3 = fsm.record_attempt_failed()
    assert st3 == RecoveryFsmState.FAILED


def test_recovery_orchestrator_fault_handling():
    backend = MockInputBackend()
    controller = CentralInputController(backend=backend)
    orchestrator = RecoveryOrchestrator(controller=controller)

    # Simulate stuck key
    controller.execute_action(ActionIntent("k_stuck", ActionType.KEY_DOWN, 0, 100, key="z"))
    assert "z" in backend.active_keys

    # Recover from stuck key fault
    success = orchestrator.handle_fault("stuck_key")
    assert success is True
    assert "z" not in backend.active_keys
    assert orchestrator.fsm.state == RecoveryFsmState.SUCCEEDED
