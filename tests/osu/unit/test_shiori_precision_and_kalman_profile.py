"""
Unit tests for Shiori Precision Upgrades:
1. PlayfieldShift (8f * scale) & NumPy Affine Matrix batch transformation.
2. Hardware baseline preservation in AdaptiveTimingCalibrator.
3. 1D Bayesian Kalman Filter Beatmap Machine Learning Profile Engine.
4. Vectorized Minimum-Jerk polynomial trajectories and batch utility scoring.
"""

import numpy as np
import pytest
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator
from src.gameplay.osu.evaluation.game_memory import OsuGameMemory
from src.gameplay.osu.core.gpu_engine import gpu_engine


def test_preprocessor_playfield_shift_and_affine_matrix():
    """Verify that playfield Y bounds incorporate 8f * scale shift and affine matrix maps coordinates accurately."""
    # 1600x900 resolution (Scale = 1.875)
    prep = PlayfieldPreprocessor(screen_width=1600, screen_height=900)
    assert prep.bounds.scale == 1.875

    # Base Y offset: (900 - 720) / 2 = 90. PlayfieldShift: 8 * 1.875 = 15.
    # Total Y must be exactly 105!
    assert prep.bounds.y == 105
    assert prep.bounds.x == 320

    # Test single point transform
    sx, sy = prep.osu_to_screen(256.0, 192.0)
    assert sx == 320 + int(256.0 * 1.875)  # 320 + 480 = 800
    assert sy == 105 + int(192.0 * 1.875)  # 105 + 360 = 465

    # Test vectorized Affine Matrix transformation (Linear Algebra)
    points_osu = np.array([
        [0.0, 0.0],
        [512.0, 384.0],
        [256.0, 192.0],
    ], dtype=np.float64)

    batch_screen = prep.osu_to_screen_batch(points_osu)
    assert batch_screen.shape == (3, 2)
    assert batch_screen[0, 0] == 320
    assert batch_screen[0, 1] == 105
    assert batch_screen[1, 0] == 320 + 960  # 1280
    assert batch_screen[1, 1] == 105 + 720  # 825
    assert batch_screen[2, 0] == 800
    assert batch_screen[2, 1] == 465


def test_timing_calibrator_baseline_hardware_preservation():
    """Verify that hardware baseline latency (e.g. 12.5ms) is preserved across dynamic resets."""
    calibrator = AdaptiveTimingCalibrator(default_latency_ms=2.0)
    assert calibrator.total_lead_time_ms == 2.0

    # Set hardware baseline (12.5ms for Windows SendInput)
    total = calibrator.set_baseline_hardware_offset_ms(12.5)
    assert total == 14.5
    assert calibrator.baseline_hardware_offset_ms == 12.5

    # Dynamic adaptive adjustment (e.g. learned +1.5ms)
    calibrator.set_adaptive_offset_ms(1.5)
    assert calibrator.adaptive_offset_ms == 1.5
    assert abs(calibrator.total_lead_time_ms - 16.0) < 0.01

    # Standard reset_offset must NOT wipe out baseline hardware offset!
    calibrator.reset_offset(reset_baseline=False)
    assert calibrator.adaptive_offset_ms == 0.0
    assert calibrator.baseline_hardware_offset_ms == 12.5
    assert calibrator.total_lead_time_ms == 14.5


def test_game_memory_kalman_machine_learning_profile(tmp_path):
    """Verify 1D Kalman Bayesian filter updates lead-time offset from hit error observations."""
    test_json = tmp_path / "test_profiles.json"
    memory = OsuGameMemory()
    # Mock profile file location
    memory.__class__.profile_file = property(lambda self: test_json)

    session_record = {
        "beatmap_id": 99999,
        "beatmap_title": "Future Cake [Extra]",
        "mode": "standard",
        "accuracy": 92.5,
        "avg_timing_error_ms": 6.0,  # Hitting 6ms late consistently
        "is_stream_heavy": True,
    }

    # First run update
    prof1 = memory.update_beatmap_profile_kalman(session_record)
    assert prof1["total_plays"] == 1
    assert prof1["learned_adaptive_offset_ms"] > 0.0  # Must nudge positive to dispatch earlier
    assert prof1["archetype"] == "stream_speed"

    # Second run with smaller error
    session_record["avg_timing_error_ms"] = 2.0
    prof2 = memory.update_beatmap_profile_kalman(session_record)
    assert prof2["total_plays"] == 2
    assert prof2["kalman_variance"] < prof1["kalman_variance"]  # Uncertainty decreases

    # Verify query
    loaded = memory.get_beatmap_profile(99999)
    assert loaded is not None
    assert loaded["total_plays"] == 2


def test_gpu_engine_vectorized_minimum_jerk_and_utility():
    """Verify Minimum-Jerk trajectory boundary conditions and NumPy batch scoring."""
    start = (100.0, 200.0)
    target = (500.0, 600.0)

    traj = gpu_engine.compute_minimum_jerk_trajectory(start, target, steps=5)
    assert len(traj) == 5

    # Final waypoint must reach target exactly
    final_x, final_y, final_u = traj[-1]
    assert abs(final_x - 500.0) < 1e-4
    assert abs(final_y - 600.0) < 1e-4
    assert abs(final_u - 1.0) < 1e-4

    # Trajectory must be monotonically advancing
    for i in range(len(traj) - 1):
        assert traj[i + 1][0] >= traj[i][0]
        assert traj[i + 1][1] >= traj[i][1]

    # Test batch utility evaluation
    candidates = [
        {"prob_hit": 0.98, "risk": 0.02, "lead_time_ms": 10.0},
        {"prob_hit": 0.70, "risk": 0.40, "lead_time_ms": 50.0},
    ]
    scores = gpu_engine.batch_evaluate_actions(candidates, current_combo=50)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # Safe hit must outscore risky hit
