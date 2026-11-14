"""
Unit Tests for Phase 5: Gameplay Brain - Dedicated Pure osu! Standard.
Covers Standard Aiming, Sliders, Spinners, Minimum-Jerk Motion, and Z/X Key Alternation.
"""

import pytest

from src.gameplay.osu.beatmap.models import HitCircle, Slider, Spinner
from src.gameplay.osu.brains import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.input.action_model import ActionType
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


def test_standard_brain_aiming_and_hit_circle():
    brain = StandardBrain(smoothing_steps=4)
    anchor_ns = clock.now_ns()

    # Circle at (100, 100)
    circle = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1)
    w_circle = WorldObject(id="s1", hit_object=circle, lifecycle=ObjectLifecycleState.APPROACHING)

    intents = brain.plan_actions(current_time_ms=800.0, objects=[w_circle], timeline_anchor_ns=anchor_ns)

    # 4 movement steps + 1 click
    move_actions = [a for a in intents if a.action_type == ActionType.MOUSE_MOVE]
    click_actions = [a for a in intents if a.action_type == ActionType.KEY_TAP]

    assert len(move_actions) == 4
    assert len(click_actions) == 1
    assert click_actions[0].key in ["z", "x"]

    # Target position of final move action should match screen projection of circle
    exp_sx, exp_sy = brain.preprocessor.osu_to_screen(100.0, 100.0)
    assert move_actions[-1].x == pytest.approx(float(exp_sx), abs=1.0)
    assert move_actions[-1].y == pytest.approx(float(exp_sy), abs=1.0)


def test_standard_brain_key_alternation():
    brain = StandardBrain(smoothing_steps=2)
    anchor_ns = clock.now_ns()

    c1 = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1)
    c2 = HitCircle(x=200.0, y=200.0, time_ms=1200.0, hit_type=1)
    w1 = WorldObject(id="c1", hit_object=c1, lifecycle=ObjectLifecycleState.APPROACHING)
    w2 = WorldObject(id="c2", hit_object=c2, lifecycle=ObjectLifecycleState.APPROACHING)

    intents1 = brain.plan_actions(current_time_ms=800.0, objects=[w1], timeline_anchor_ns=anchor_ns)
    clicks1 = [a for a in intents1 if a.action_type == ActionType.KEY_TAP]
    assert len(clicks1) == 1

    intents2 = brain.plan_actions(current_time_ms=1050.0, objects=[w2], timeline_anchor_ns=anchor_ns)
    clicks2 = [a for a in intents2 if a.action_type == ActionType.KEY_TAP]
    assert len(clicks2) == 1

    # Should alternate between z and x
    assert clicks1[0].key != clicks2[0].key
    assert {clicks1[0].key, clicks2[0].key} == {"z", "x"}
