"""
Input Execution Domain for Shiori AI osu! Gameplay.
Exports ActionIntent, ActionType, CentralInputQueue, CentralInputController, and EmergencyStopManager.
"""

from src.gameplay.osu.input.action_model import (
    ActionIntent,
    ActionType,
    InputStateStatus,
)
from src.gameplay.osu.input.backends import (
    BaseInputBackend,
    MockInputBackend,
    Win32SendInputBackend,
)
from src.gameplay.osu.input.controller import (
    CentralInputController,
    input_controller,
)
from src.gameplay.osu.input.emergency import (
    EmergencyStopManager,
    emergency_stop_manager,
)
from src.gameplay.osu.input.queue import CentralInputQueue, input_queue

__all__ = [
    "ActionType",
    "InputStateStatus",
    "ActionIntent",
    "CentralInputQueue",
    "input_queue",
    "CentralInputController",
    "input_controller",
    "EmergencyStopManager",
    "emergency_stop_manager",
    "BaseInputBackend",
    "Win32SendInputBackend",
    "MockInputBackend",
]
