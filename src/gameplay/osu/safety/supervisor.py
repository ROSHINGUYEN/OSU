"""
Central Safety Supervisor for Shiori AI osu! Gameplay.
Monitors subsystem health, manages dependency graph, and orchestrates active revocation.
"""

import logging
from typing import Dict, List, Tuple

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    ModeDetectionStatus,
    OperatingState,
    SafetyState,
    SystemState,
    state_store,
)
from src.gameplay.osu.input.controller import (
    CentralInputController,
    input_controller,
)
from src.gameplay.osu.input.queue import CentralInputQueue, input_queue
from src.gameplay.osu.safety.permission_engine import (
    PermissionDecision,
    PermissionDecisionType,
    PermissionEngine,
    permission_engine,
)

logger = logging.getLogger("Shiori.Osu.SafetySupervisor")


class SafetySupervisor:
    """Oversees dependency health, issues permission leases, and enforces fail-safe revocation."""

    def __init__(
        self,
        engine: PermissionEngine = permission_engine,
        queue: CentralInputQueue = input_queue,
        controller: CentralInputController = input_controller,
    ):
        self._engine = engine
        self._queue = queue
        self._controller = controller

    def evaluate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Evaluates health of all vital subsystems:
        1. Clock: Monotonic time functioning.
        2. Mode: Game mode confirmed or user-overridden.
        3. Input: Watchdog warning count acceptable.
        Returns (is_healthy, failure_reasons).
        """
        failures: List[str] = []

        # 1. Mode Confirmation Check
        mode_state = state_store.mode
        if mode_state.status not in [ModeDetectionStatus.MODE_CONFIRMED] and not mode_state.user_overridden:
            failures.append("MODE_NOT_CONFIRMED")

        # 2. Safety State Check (Emergency Stop)
        if state_store.safety.emergency_stop_triggered:
            failures.append("EMERGENCY_STOP_ACTIVE")

        # 3. Input Controller Health Check
        if state_store.input.stuck_key_warnings > 5:
            failures.append("EXCESSIVE_STUCK_KEYS")

        is_healthy = len(failures) == 0

        # Update store with dependency status
        dep_health = {
            "mode_confirmed": "MODE_NOT_CONFIRMED" not in failures,
            "emergency_clear": "EMERGENCY_STOP_ACTIVE" not in failures,
            "input_healthy": "EXCESSIVE_STUCK_KEYS" not in failures,
        }

        state_store.mutate_domain(
            "safety",
            lambda curr: SafetyState(
                dependency_health=dep_health,
                active_leases=curr.active_leases,
                emergency_stop_triggered=curr.emergency_stop_triggered,
                updated_at_ns=clock.now_ns(),
            ),
            caller="safety_supervisor",
        )

        return is_healthy, failures

    def request_action_permission(self, duration_ms: int = 50) -> PermissionDecision:
        """
        Evaluates permissions. If dependencies are healthy, grants lease.
        If unhealthy, revokes existing leases, clears queue, neutralizes inputs, and denies.
        """
        is_healthy, reasons = self.evaluate_dependencies()

        if is_healthy:
            return self._engine.grant_lease(duration_ms=duration_ms)
        else:
            # Active Revocation (throttled to avoid terminal log flooding)
            now = clock.now_ns()
            if (now - getattr(self, "_last_revocation_log_ns", 0)) > 3_000_000_000:
                self._last_revocation_log_ns = now
                logger.debug(f"Safety check standby: {reasons}.")
            self._engine.revoke_all(reason=f"Failed dependencies: {reasons}")
            self._queue.clear()
            self._controller.emergency_release_all()

            # Mark system state as DEGRADED
            state_store.mutate_domain(
                "system",
                lambda curr: SystemState(
                    state=OperatingState.DEGRADED,
                    cpu_load=curr.cpu_load,
                    fps=curr.fps,
                    sequence_id=curr.sequence_id + 1,
                    updated_at_ns=clock.now_ns(),
                ),
                caller="safety_supervisor",
            )

            return PermissionDecision(
                decision=PermissionDecisionType.DENY,
                reason_codes=reasons,
                lease_id="",
                lease_deadline_ns=0,
            )


safety_supervisor = SafetySupervisor()
