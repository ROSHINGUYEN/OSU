"""
Emergency Stop and Active Neutralization Engine.
Instantly revokes all permissions, clears the input queue, and neutralizes OS inputs.
"""

import logging

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.event_bus import Event, event_bus
from src.gameplay.osu.core.state_store import (
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

logger = logging.getLogger("Shiori.Osu.EmergencyStop")


class EmergencyStopManager:
    """Coordinates immediate cut-off and neutralization of all OS input channels."""

    def __init__(
        self,
        controller: CentralInputController = input_controller,
        queue: CentralInputQueue = input_queue,
    ):
        self._controller = controller
        self._queue = queue
        self._triggered = False

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    def trigger(self, reason: str = "Emergency stop requested by user or watchdog") -> None:
        """Execute emergency stop sequence."""
        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")
        self._triggered = True

        # 1. Clear input queue
        self._queue.clear()

        # 2. Release all keys on OS level
        self._controller.emergency_release_all()

        # 3. Update safety state
        state_store.mutate_domain(
            "safety",
            lambda curr: SafetyState(
                dependency_health=curr.dependency_health,
                active_leases={},
                emergency_stop_triggered=True,
                updated_at_ns=clock.now_ns(),
            ),
            caller="emergency_stop",
        )

        # 4. Set system state to SAFE_STOP
        state_store.mutate_domain(
            "system",
            lambda curr: SystemState(
                state=OperatingState.SAFE_STOP,
                cpu_load=curr.cpu_load,
                fps=curr.fps,
                sequence_id=curr.sequence_id + 1,
                updated_at_ns=clock.now_ns(),
            ),
            caller="system",
        )

        # 5. Broadcast event
        event_bus.publish_sync(
            Event(
                event_type="emergency_stop_activated",
                payload={"reason": reason},
                source="emergency_stop",
            )
        )

    def reset(self) -> None:
        """Reset emergency stop state after inspection."""
        self._triggered = False
        state_store.mutate_domain(
            "safety",
            lambda curr: SafetyState(
                dependency_health=curr.dependency_health,
                active_leases=curr.active_leases,
                emergency_stop_triggered=False,
                updated_at_ns=clock.now_ns(),
            ),
            caller="emergency_stop",
        )
        logger.info("Emergency stop has been reset.")


emergency_stop_manager = EmergencyStopManager()
