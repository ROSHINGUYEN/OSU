"""
Unit tests for Shiori Anti-Ban & Biometric Humanization Engine.
"""

import pytest
from src.gameplay.osu.safety.humanizer import AntiBanHumanizer, HumanizerConfig


def test_humanizer_toggle():
    h = AntiBanHumanizer(HumanizerConfig(enabled=True))
    assert h.is_enabled is True
    h.set_enabled(False)
    assert h.is_enabled is False


def test_hit_timing_jitter_within_hit300_window():
    h = AntiBanHumanizer(HumanizerConfig(enabled=True, timing_sigma_ms=3.0))
    # Test for OD 8, 9, 10
    for od in [8.0, 9.0, 10.0]:
        hit_300_window_ms = 80.0 - 6.0 * od
        for i in range(50):
            jitter_ns = h.calculate_hit_timing_jitter_ns(f"hit_{i}_{od}", od=od)
            jitter_ms = abs(jitter_ns / 1_000_000.0)
            # Must strictly stay within Hit 300 window margin
            assert jitter_ms < hit_300_window_ms, f"Jitter {jitter_ms}ms exceeded hit 300 window {hit_300_window_ms}ms"


def test_spatial_click_dispersion():
    h = AntiBanHumanizer(HumanizerConfig(enabled=True, spatial_dispersion_px=4.0))
    center_x, center_y = 500, 400
    radius_px = 32.0

    for i in range(30):
        hx, hy = h.humanize_click_position(center_x, center_y, radius_px, f"circle_{i}")
        dist = ((hx - center_x) ** 2 + (hy - center_y) ** 2) ** 0.5
        # Must be dispersed but stay within circle boundary
        assert dist <= radius_px * 0.35, f"Dispersion {dist}px exceeded safe bound"


def test_rank_surge_fatigue_pacing():
    h = AntiBanHumanizer(HumanizerConfig(enabled=True, max_consecutive_fc=3, fatigue_enabled=True))
    # Initial state
    assert h._consecutive_fc == 0
    assert h._current_fatigue_multiplier == 1.0

    # Simulate 3 consecutive Full Combos
    h.on_map_completed(combo=500, max_combo=500, accuracy=99.8)
    h.on_map_completed(combo=700, max_combo=700, accuracy=100.0)
    assert h._consecutive_fc == 2
    assert h._current_fatigue_multiplier == 1.0

    # 3rd and 4th consecutive FC triggers fatigue pacer
    h.on_map_completed(combo=800, max_combo=800, accuracy=99.9)
    assert h._consecutive_fc == 3
    assert h._current_fatigue_multiplier > 1.0

    h.on_map_completed(combo=900, max_combo=900, accuracy=99.7)
    assert h._consecutive_fc == 4
    assert h._current_fatigue_multiplier >= 1.25


def test_key_hold_duration_within_range():
    h = AntiBanHumanizer(HumanizerConfig(enabled=True, min_hold_duration_ms=25.0, max_hold_duration_ms=50.0))
    for i in range(20):
        hold_ns = h.calculate_key_hold_duration_ns(bpm=180.0, hit_id=f"note_{i}")
        hold_ms = hold_ns / 1_000_000.0
        assert 15.0 <= hold_ms <= 65.0, f"Hold time {hold_ms}ms outside expected range"
