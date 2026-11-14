"""
Pytest configuration and test cleanup fixture for osu! test suite.
Ensures test artifacts do not clutter disk or memory.
"""

import shutil
from pathlib import Path
import pytest

from src.gameplay.osu.evaluation.telemetry import telemetry_logger
from src.gameplay.osu.evaluation.debug_logger import visual_debugger


@pytest.fixture(autouse=True, scope="session")
def cleanup_osu_test_artifacts():
    """Clean up test telemetry and debug outputs before and after the test session."""
    # Pre-test cleanup
    telemetry_logger.clear_all_sessions(prefix="session_test_")
    visual_debugger.clear_debug_files()

    yield

    # Post-test cleanup
    telemetry_logger.stop_session()
    telemetry_logger.clear_all_sessions(prefix="session_test_")
    visual_debugger.clear_debug_files()

    # Remove test debug dir if created during tests
    test_dir = Path("data/osu_debug_test")
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)
