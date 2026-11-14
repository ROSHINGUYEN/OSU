"""
Base Gameplay Brain Interface for Shiori AI osu!.
Defines candidate action generation across all game modes.
"""

from abc import ABC, abstractmethod
from typing import List

from src.gameplay.osu.core.state_store import GameMode
from src.gameplay.osu.input.action_model import ActionIntent
from src.gameplay.osu.world.object_lifecycle import WorldObject


class BaseGameplayBrain(ABC):
    """Abstract gameplay intelligence producing ActionIntents from WorldObjects."""

    def __init__(self, mode: GameMode):
        self._mode = mode

    @property
    def mode(self) -> GameMode:
        return self._mode

    @abstractmethod
    def plan_actions(
        self,
        current_time_ms: float,
        objects: List[WorldObject],
        timeline_anchor_ns: int,
    ) -> List[ActionIntent]:
        """
        Evaluate upcoming world objects within planning horizon and generate ActionIntents.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal brain states (e.g. key alternation, cursor smoothing)."""
        pass
