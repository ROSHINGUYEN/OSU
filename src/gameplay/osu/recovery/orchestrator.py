"""
Subsystem Recovery Orchestrator for Shiori AI osu! Gameplay.
Coordinates targeted recovery actions for capture, timing, input, and mode faults.
"""

import logging

from src.gameplay.osu.input.controller import (
    CentralInputController,
    input_controller,
)
from src.gameplay.osu.perception.capture import CaptureService, capture_service
from src.gameplay.osu.perception.mode_detector import (
    ModeDetectionService,
    mode_detection_service,
)
from src.gameplay.osu.recovery.recovery_fsm import RecoveryFsm
from src.gameplay.osu.timing.calibration import (
    AdaptiveTimingCalibrator,
    timing_calibrator,
)

logger = logging.getLogger("Shiori.Osu.Recovery")


class RecoveryOrchestrator:
    """Automates mitigation and self-healing for transient subsystem failures."""

    def __init__(
        self,
        fsm: RecoveryFsm = None,
        controller: CentralInputController = input_controller,
        capture: CaptureService = capture_service,
        mode_detector: ModeDetectionService = mode_detection_service,
        calibrator: AdaptiveTimingCalibrator = timing_calibrator,
    ):
        self.fsm = fsm or RecoveryFsm()
        self._controller = controller
        self._capture = capture
        self._mode_detector = mode_detector
        self._calibrator = calibrator

    def handle_fault(self, fault_type: str) -> bool:
        """Executes recovery action for a recognized failure type."""
        logger.info(f"Initiating recovery for fault '{fault_type}'...")
        if not self.fsm.request_recovery(fault_type):
            return False

        self.fsm.start_recovering()
        success = False

        try:
            if fault_type == "stuck_key":
                self._controller.emergency_release_all()
                success = True

            elif fault_type == "clock_drift":
                self._calibrator.reset_offset()
                success = True

            elif fault_type == "capture_freeze":
                self._capture.stop()
                success = self._capture.start()

            elif fault_type == "mode_conflict":
                # Re-evaluate mode with neutral defaults
                self._mode_detector.evaluate_mode()
                success = True

            else:
                logger.warning(f"Unknown fault type: '{fault_type}'. Resetting state safely.")
                self._controller.emergency_release_all()
                success = True

        except Exception as e:
            logger.error(f"Error during recovery of {fault_type}: {e}", exc_info=True)
            success = False

        self.fsm.start_verifying()
        if success:
            self.fsm.mark_succeeded()
            logger.info(f"Recovery for '{fault_type}' succeeded.")
            return True
        else:
            self.fsm.record_attempt_failed()
            logger.error(f"Recovery for '{fault_type}' failed.")
            return False


recovery_orchestrator = RecoveryOrchestrator()
