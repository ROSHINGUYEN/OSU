"""
Recovery State Machine for Shiori AI osu! Gameplay.
Guarantees bounded retries, backoff, and state verification before resuming gameplay.
"""

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    RecoveryFsmState,
    RecoveryState,
    state_store,
)


class RecoveryFsm:
    """Finite State Machine tracking the progress of fault recovery."""

    def __init__(self, max_retries: int = 3):
        self._state = RecoveryFsmState.IDLE
        self._retry_count = 0
        self._max_retries = max_retries
        self._active_fault = ""

    @property
    def state(self) -> RecoveryFsmState:
        return self._state

    @property
    def retry_count(self) -> int:
        return self._retry_count

    def request_recovery(self, fault_type: str) -> bool:
        """Initiate recovery sequence for a specified fault."""
        if self._state not in [RecoveryFsmState.IDLE, RecoveryFsmState.FAILED, RecoveryFsmState.SUCCEEDED]:
            return False  # Already recovering

        self._state = RecoveryFsmState.REQUESTED
        self._active_fault = fault_type
        self._retry_count = 0
        self._sync_to_store()
        return True

    def start_recovering(self) -> None:
        self._state = RecoveryFsmState.RECOVERING
        self._sync_to_store()

    def record_attempt_failed(self) -> RecoveryFsmState:
        """Handle a failed attempt. Increments retry counter or transitions to FAILED."""
        self._retry_count += 1
        if self._retry_count > self._max_retries:
            self._state = RecoveryFsmState.FAILED
        else:
            self._state = RecoveryFsmState.BACKOFF
        self._sync_to_store()
        return self._state

    def start_verifying(self) -> None:
        self._state = RecoveryFsmState.VERIFYING
        self._sync_to_store()

    def mark_succeeded(self) -> None:
        self._state = RecoveryFsmState.SUCCEEDED
        self._sync_to_store()

    def reset_to_idle(self) -> None:
        self._state = RecoveryFsmState.IDLE
        self._retry_count = 0
        self._active_fault = ""
        self._sync_to_store()

    def _sync_to_store(self) -> None:
        state_store.mutate_domain(
            "recovery",
            lambda curr: RecoveryState(
                state=self._state,
                retry_count=self._retry_count,
                active_fault=self._active_fault,
                updated_at_ns=clock.now_ns(),
            ),
            caller="recovery_orchestrator",
        )
