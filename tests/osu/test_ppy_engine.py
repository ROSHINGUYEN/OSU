"""
Unit tests for ppy/osu Engine Bridge & Algorithms.
Validates hit window calculations, playfield geometry, and key mappings against ppy/osu source.
"""

import pytest
from src.gameplay.osu.integrations.ppy_engine import (
    calculate_taiko_hit_windows,
    compute_taiko_target_center_x,
    get_taiko_action_key_bindings,
)


def test_taiko_hit_windows_od5():
    hw = calculate_taiko_hit_windows(5.0)
    assert hw["great"] == 34.5
    assert hw["ok"] == 79.5
    assert hw["miss"] == 94.5


def test_taiko_hit_windows_od0():
    hw = calculate_taiko_hit_windows(0.0)
    assert hw["great"] == 49.5
    assert hw["ok"] == 119.5
    assert hw["miss"] == 134.5


def test_taiko_hit_windows_od10():
    hw = calculate_taiko_hit_windows(10.0)
    assert hw["great"] == 19.5
    assert hw["ok"] == 49.5
    assert hw["miss"] == 69.5


def test_taiko_target_center_x():
    x_720p = compute_taiko_target_center_x(1280, 720)
    assert 245.0 <= x_720p <= 255.0


def test_taiko_action_key_bindings():
    bindings = get_taiko_action_key_bindings()
    assert bindings["LeftCentre"] == ("f", 0x21)
    assert bindings["RightCentre"] == ("j", 0x24)
    assert bindings["LeftRim"] == ("d", 0x20)
    assert bindings["RightRim"] == ("k", 0x25)
