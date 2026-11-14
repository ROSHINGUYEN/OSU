"""
Game Mode Detection Service for Shiori AI osu!.
Combines Beatmap metadata, UI evidence, vision heuristics, and user overrides.
"""

import logging
from typing import Optional

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.event_bus import Event, event_bus
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    state_store,
)

logger = logging.getLogger("Shiori.Osu.ModeDetector")


class ModeDetectionService:
    """Detects and confirms the active game mode with conflict resolution."""

    def __init__(self):
        self._user_override: Optional[GameMode] = None

    def set_user_override(self, mode: Optional[GameMode]) -> None:
        """Allow user (Ba Roshi) to force a specific game mode."""
        self._user_override = mode
        if mode:
            self._update_store(mode, ModeDetectionStatus.MODE_CONFIRMED, confidence=1.0, overridden=True)
            logger.info(f"Game mode manually overridden to {mode.value}.")
        else:
            self._update_store(GameMode.UNKNOWN, ModeDetectionStatus.UNKNOWN, confidence=0.0, overridden=False)
            logger.info("Game mode manual override cleared.")

    def clear_user_override(self) -> None:
        """Clear any user override, returning mode detection to automatic."""
        self.set_user_override(None)

    def evaluate_mode(
        self,
        beatmap_mode: Optional[GameMode] = None,
        vision_detected_mode: Optional[GameMode] = None,
        confidence: float = 0.9,
    ) -> ModeState:
        """Evaluate mode from all available sources and update CentralStateStore."""
        if self._user_override is not None:
            return self._update_store(
                self._user_override,
                ModeDetectionStatus.MODE_CONFIRMED,
                confidence=1.0,
                overridden=True,
            )

        # Case 1: Both Beatmap and Vision agree
        if beatmap_mode and vision_detected_mode:
            if beatmap_mode == vision_detected_mode:
                return self._update_store(
                    beatmap_mode,
                    ModeDetectionStatus.MODE_CONFIRMED,
                    confidence=min(1.0, confidence + 0.1),
                    overridden=False,
                )
            else:
                logger.warning(
                    f"Mode conflict detected: Beatmap={beatmap_mode.value} vs Vision={vision_detected_mode.value}"
                )
                event_bus.publish_sync(
                    Event(
                        event_type="mode_conflict",
                        payload={"beatmap": beatmap_mode.value, "vision": vision_detected_mode.value},
                        source="mode_detector",
                    )
                )
                return self._update_store(
                    GameMode.STANDARD,
                    ModeDetectionStatus.MODE_CONFIRMED,
                    confidence=0.98,
                    overridden=False,
                )

        # Case 2: Only Beatmap is available
        if beatmap_mode:
            return self._update_store(
                beatmap_mode,
                ModeDetectionStatus.MODE_CONFIRMED,
                confidence=0.95,
                overridden=False,
            )

        # Case 3: Only Vision is available
        if vision_detected_mode:
            return self._update_store(
                vision_detected_mode,
                ModeDetectionStatus.MODE_CONFIRMED,
                confidence=confidence,
                overridden=False,
            )

        # Case 4: Neither is available
        return self._update_store(
            GameMode.UNKNOWN,
            ModeDetectionStatus.UNKNOWN,
            confidence=0.0,
            overridden=False,
        )

    def _update_store(
        self,
        mode: GameMode,
        status: ModeDetectionStatus,
        confidence: float,
        overridden: bool,
    ) -> ModeState:
        new_state = state_store.mutate_domain(
            "mode",
            lambda curr: ModeState(
                mode=mode,
                status=status,
                confidence=confidence,
                user_overridden=overridden,
                updated_at_ns=clock.now_ns(),
            ),
            caller="mode_detector",
        )
        return new_state


mode_detection_service = ModeDetectionService()
