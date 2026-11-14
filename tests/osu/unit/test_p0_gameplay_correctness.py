"""
Unit & Regression Tests for P0 Gameplay Correctness Fixes.

Covers:
- G-01: Slider repeat-count / duration / geometry consistency (P0 4.1-4.3)
- G-02: Overlapping / same-coordinate circles keep distinct identities (P0 4.4)
- G-03: Atomic per-target enqueue; partial enqueue never marks a target scheduled (P0 8.3)
- G-04: Queue waypoint coalescing never merges across different targets/sequences (P0 8.2)
- G-05: Hit-evaluation timing delta measured against intended hit time, not dispatch time (P0 5.1, 9)
- G-06: get_brain explicit mode mapping; unknown/unsupported mode returns None (P1 11)
- G-07: Enforced object lifecycle transitions (P0 9)
- G-08: Planner skips past-deadline / stale targets instead of scheduling them (P0 5.2, 6.3)
"""

import math

import pytest

from src.gameplay.osu.beatmap.models import (
    BeatmapData,
    CurveType,
    HitCircle,
    HitObjectFlag,
    Slider,
)
from src.gameplay.osu.beatmap.parser import BeatmapParser
from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    ModeState,
    state_store,
)
from src.gameplay.osu.evaluation.telemetry import telemetry_logger
from src.gameplay.osu.evaluation.feedback import feedback_evaluator
from src.gameplay.osu.input.action_model import (
    ActionIntent,
    ActionType,
    CursorOwnerState,
    InputStateStatus,
)
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.controller import CentralInputController
from src.gameplay.osu.input.queue import CentralInputQueue
from src.gameplay.osu.math.slider_curves import calculate_slider_path
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.planning.planner import RollingHorizonPlanner
from src.gameplay.osu.world.object_lifecycle import (
    ObjectLifecycleState,
    WorldObject,
)
from src.gameplay.osu.world.world_state import WorldStateManager


# ---------------------------------------------------------------------------
# Sample beatmap fixtures
# ---------------------------------------------------------------------------

SIMPLE_OSU = """osu file format v14

[General]
Mode: 0

[Metadata]
Title: Slider Repeat Test
Artist: Shiori
Creator: Roshi
Version: Test

[Difficulty]
HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 8
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1

[TimingPoints]
1000,500,4,1,0,100,1,0

[HitObjects]
100,100,1000,2,0,B|200:100,1,100,0:0|0:0,0:0:0:0:
100,100,2000,2,0,B|200:100,2,100,0:0|0:0,0:0:0:0:
"""


# ---------------------------------------------------------------------------
# G-01: Slider repeat-count / duration / geometry consistency (P0 4.1-4.3)
# ---------------------------------------------------------------------------

def test_slider_repeat_duration_consistency():
    """repeat_count=2 must take exactly 2x the duration of repeat_count=1
    (same pixel length and timing); span count == repeat count."""
    beatmap = BeatmapParser.parse_string(SIMPLE_OSU)
    s_one = beatmap.hit_objects[0]
    s_two = beatmap.hit_objects[1]

    assert isinstance(s_one, Slider) and isinstance(s_two, Slider)
    assert s_one.repeat_count == 1
    assert s_two.repeat_count == 2
    assert s_one.pixel_length == s_two.pixel_length == 100.0

    # Same pixel length / timing -> double the repeats doubles the duration.
    assert s_two.duration_ms == pytest.approx(2.0 * s_one.duration_ms, rel=1e-6)
    assert s_two.end_time_ms == pytest.approx(s_two.time_ms + s_two.duration_ms)
    assert s_one.end_time_ms == pytest.approx(s_one.time_ms + s_one.duration_ms)

    # Span count equals repeat count; path is consistent across repeats.
    path_one = calculate_slider_path(s_one)
    path_two = calculate_slider_path(s_two)
    assert len(path_one) >= 2
    assert len(path_two) >= 2
    assert path_two[0] == path_one[0]  # same head


# ---------------------------------------------------------------------------
# G-02: Overlapping / same-coordinate circles keep distinct identities (P0 4.4)
# ---------------------------------------------------------------------------

def test_same_coordinate_circles_not_merged():
    """Objects at identical coordinates but different hit times must remain
    distinct, distinguishable timeline entries (never merged by position)."""
    c1 = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1)
    c2 = HitCircle(x=256.0, y=192.0, time_ms=1050.0, hit_type=1)

    assert c1 is not c2
    assert c1.time_ms != c2.time_ms

    timeline = HitObjectTimeline([c1, c2])
    objs = timeline.get_objects_in_window(900.0, 1100.0)
    assert len(objs) == 2
    assert {o.time_ms for o in objs} == {1000.0, 1050.0}
    assert objs[0] is not objs[1]


# ---------------------------------------------------------------------------
# G-03: Atomic per-target enqueue (P0 8.3)
# ---------------------------------------------------------------------------

def test_atomic_enqueue_full_sequence_commits_and_schedules():
    """When the queue has capacity, a slider's entire sequence
    (down + waypoints + up) is committed and the target is ACTION_SCHEDULED."""
    queue = CentralInputQueue(max_size=1000)
    world_mgr = WorldStateManager()
    prep = PlayfieldPreprocessor(1280, 720)
    planner = RollingHorizonPlanner(
        horizon_ms=1000.0, commit_window_ms=200.0,
        queue=queue, world_manager=world_mgr, preprocessor=prep,
    )
    anchor_ns = clock.now_ns()

    slider = Slider(
        x=100.0, y=100.0, time_ms=200.0, hit_type=HitObjectFlag.SLIDER,
        curve_points=[(200.0, 150.0)], repeat_count=1,
        duration_ms=400.0, end_time_ms=600.0,
    )
    w = WorldObject("sl1", slider, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w)

    committed = planner.plan_cycle(
        current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD
    )

    assert len(committed) >= 1, "Expected slider sequence to be committed"
    assert len(queue) == len(committed), "All committed actions must be enqueued"
    assert world_mgr.get_object("sl1").lifecycle == ObjectLifecycleState.ACTION_SCHEDULED


def test_atomic_enqueue_queue_full_rejects_whole_sequence():
    """When the queue cannot hold the full target sequence, push_group rejects
    the whole group: nothing is enqueued and the target is NOT scheduled
    (left for replanning on the next cycle)."""
    queue = CentralInputQueue(max_size=2)  # far too small for a slider sequence
    world_mgr = WorldStateManager()
    prep = PlayfieldPreprocessor(1280, 720)
    planner = RollingHorizonPlanner(
        horizon_ms=1000.0, commit_window_ms=200.0,
        queue=queue, world_manager=world_mgr, preprocessor=prep,
    )
    anchor_ns = clock.now_ns()

    slider = Slider(
        x=100.0, y=100.0, time_ms=200.0, hit_type=HitObjectFlag.SLIDER,
        curve_points=[(200.0, 150.0)], repeat_count=1,
        duration_ms=400.0, end_time_ms=600.0,
    )
    w = WorldObject("sl2", slider, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w)

    committed = planner.plan_cycle(
        current_time_ms=0.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD
    )

    assert len(committed) == 0, "No action should be committed when the sequence cannot be atomically enqueued"
    assert len(queue) == 0, "No partial enqueue is allowed"
    assert world_mgr.get_object("sl2").lifecycle != ObjectLifecycleState.ACTION_SCHEDULED



# ---------------------------------------------------------------------------
# G-04: Queue coalescing restricted to a single target sequence (P0 8.2)
# ---------------------------------------------------------------------------

def test_queue_coalescing_never_merges_across_different_targets():
    """Waypoints from Slider A must never replace waypoints from Slider B even
    though both share the SLIDER_FOLLOW cursor owner. Each target collapses to
    its own newest overdue waypoint; no cross-target contamination."""
    queue = CentralInputQueue()
    base_ns = 1_000_000_000

    for i in range(1, 4):
        queue.push(ActionIntent(
            action_id=f"act_s1_slide_{i}", action_type=ActionType.MOUSE_MOVE,
            execute_at_ns=base_ns + i * 10_000_000,
            deadline_ns=base_ns + i * 10_000_000 + 100_000_000,
            target_id="s1", cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
            x=float(i), y=float(i), priority=1,
        ))
    for i in range(1, 4):
        queue.push(ActionIntent(
            action_id=f"act_s2_slide_{i}", action_type=ActionType.MOUSE_MOVE,
            execute_at_ns=base_ns + i * 10_000_000,
            deadline_ns=base_ns + i * 10_000_000 + 100_000_000,
            target_id="s2", cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
            x=float(100 + i), y=float(100 + i), priority=1,
        ))

    pops = []
    while True:
        popped = queue.pop_due(base_ns + 42_000_000)
        if popped is None:
            break
        pops.append(popped)

    # Fixed behavior: every waypoint is returned under its own target identity;
    # nothing is silently dropped or replaced by another target's waypoint.
    # (Under the old buggy coalescing, a single pop would swallow the whole set.)
    assert len(pops) == 6, f"Expected all 6 waypoints returned, got {len(pops)}: {pops}"
    expected = {
        (f"act_s1_slide_{i}", "s1") for i in range(1, 4)
    } | {(f"act_s2_slide_{i}", "s2") for i in range(1, 4)}
    assert {(p.action_id, p.target_id) for p in pops} == expected

    # Queue must now be empty (no waypoint silently dropped or duplicated).
    assert len(queue) == 0


def test_queue_coalescing_still_collapses_same_target_waypoints():
    """Same-target overdue waypoints still collapse to the newest (existing behavior preserved)."""
    queue = CentralInputQueue()
    base_ns = 1_000_000_000
    for i in range(1, 5):
        queue.push(ActionIntent(
            action_id=f"act_t1_slide_{i}", action_type=ActionType.MOUSE_MOVE,
            execute_at_ns=base_ns + i * 10_000_000,
            deadline_ns=base_ns + i * 10_000_000 + 100_000_000,
            target_id="t1", cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
            x=float(i), y=float(i), priority=1,
        ))
    popped = queue.pop_due(base_ns + 42_000_000)
    assert popped is not None
    assert popped.action_id == "act_t1_slide_4"
    assert popped.x == 4.0 and popped.y == 4.0



# ---------------------------------------------------------------------------
# G-05: Hit-evaluation delta vs intended hit time, not dispatch time (P0 5.1, 9)
# ---------------------------------------------------------------------------

def test_hit_delta_measured_against_intended_hit_time():
    """The timing delta fed to hit evaluation must be (execution - intended hit time),
    never (execution - dispatch time). Dispatch is lead-compensated, so measuring
    against it would fabricate a near-zero delta / false '300' for every note."""
    app = None
    try:
        app = OsuGameplayApp()
        app.controller.set_backend(MockInputBackend())
        state_store.mutate_domain(
            "mode",
            lambda m: m.__class__(
                mode=GameMode.STANDARD,
                status=ModeDetectionStatus.MODE_CONFIRMED,
                user_overridden=False,
            ),
            caller="system",
        )
        circle = HitCircle(x=256.0, y=192.0, time_ms=200.0, hit_type=1)
        app._timeline = HitObjectTimeline([circle])

        captured: dict = {}
        orig_evaluate = feedback_evaluator.evaluate_hit

        def spy(obj_id, delta_ms, od, mode, confidence=1.0, source=None):
            captured["delta_ms"] = delta_ms
            captured["obj_id"] = obj_id
            return orig_evaluate(obj_id, delta_ms, od, mode, confidence, source)

        feedback_evaluator.evaluate_hit = spy  # type: ignore[method-assign]
        try:
            app.start(session_id="test_hit_delta", run_background_thread=False)
            # Step 5ms before the note hits (200ms). Dispatch happens at 200 - lead(~14.5) = ~185.5ms,
            # so the action is due at step time 195ms and will be popped/executed.
            app.step(simulated_time_ms=195.0)
        finally:
            feedback_evaluator.evaluate_hit = orig_evaluate  # type: ignore[method-assign]
            app.stop()

        assert "delta_ms" in captured, "No hit evaluation occurred; the note was not dispatched"
        # Delta = effective now (195ms) - intended hit time (200ms) = -5ms.
        # It must NOT be ~0 (which would indicate measurement against dispatch time).
        assert abs(captured["delta_ms"] - (-5.0)) < 2.0, \
            f"Expected delta ~ -5ms (execution vs hit time), got {captured['delta_ms']}ms"
    finally:
        telemetry_logger.clear_all_sessions(prefix="session_test_")


# ---------------------------------------------------------------------------
# G-06: get_brain explicit mode mapping (P1 11)
# ---------------------------------------------------------------------------

def test_get_brain_explicit_mode_mapping():
    """get_brain returns the Standard brain for STANDARD and None for
    UNKNOWN; it never silently falls back to Standard for an unknown mode."""
    planner = RollingHorizonPlanner()
    assert planner.get_brain(GameMode.STANDARD) is not None
    assert planner.get_brain(GameMode.UNKNOWN) is None
    assert planner.get_brain(None) is None

    # Planning with an unknown/unsupported mode must safely abort rather than
    # falling back to Standard.
    wm = WorldStateManager()
    planner2 = RollingHorizonPlanner(world_manager=wm)
    obj = WorldObject("c", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1))
    wm.add_or_update(obj)
    intents = planner2.plan_cycle(
        current_time_ms=50.0, timeline_anchor_ns=1_000_000_000, mode=GameMode.UNKNOWN
    )
    assert intents == []



# ---------------------------------------------------------------------------
# G-07: Enforced object lifecycle transitions (P0 9)
# ---------------------------------------------------------------------------

def test_lifecycle_valid_transition_applied():
    w = WorldObject("o1", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1))
    assert w.lifecycle == ObjectLifecycleState.PREDICTED
    assert w.transition_to(ObjectLifecycleState.ACTION_SCHEDULED) is True
    assert w.lifecycle == ObjectLifecycleState.ACTION_SCHEDULED


def test_lifecycle_invalid_transition_rejected():
    w = WorldObject("o1", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1))
    # Reach a terminal state through valid transitions first.
    assert w.transition_to(ObjectLifecycleState.ACTION_SCHEDULED) is True
    assert w.transition_to(ObjectLifecycleState.HIT_CONFIRMED) is True
    assert w.lifecycle == ObjectLifecycleState.HIT_CONFIRMED
    # Terminal states reject any further transition.
    assert w.transition_to(ObjectLifecycleState.ACTION_SCHEDULED) is False
    assert w.transition_to(ObjectLifecycleState.MISS_CONFIRMED) is False
    assert w.lifecycle == ObjectLifecycleState.HIT_CONFIRMED


def test_lifecycle_expired_and_cancelled_allowed_from_nonterminal():
    w = WorldObject("o1", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1))
    w.transition_to(ObjectLifecycleState.ACTION_SCHEDULED)
    assert w.transition_to(ObjectLifecycleState.EXPIRED) is True
    assert w.lifecycle == ObjectLifecycleState.EXPIRED
    # EXPIRED is terminal.
    assert w.transition_to(ObjectLifecycleState.CANCELLED) is False

    w2 = WorldObject("o2", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1))
    w2.transition_to(ObjectLifecycleState.APPROACHING)
    assert w2.transition_to(ObjectLifecycleState.CANCELLED) is True
    assert w2.lifecycle == ObjectLifecycleState.CANCELLED


def test_world_manager_transition_state_returns_bool():
    mgr = WorldStateManager()
    mgr.add_or_update(WorldObject("o1", HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1)))
    assert mgr.transition_state("o1", ObjectLifecycleState.APPROACHING) is True
    assert mgr.get_object("o1").lifecycle == ObjectLifecycleState.APPROACHING
    assert mgr.transition_state("does_not_exist", ObjectLifecycleState.APPROACHING) is False


# ---------------------------------------------------------------------------
# G-08: Planner skips past-deadline / stale targets (P0 5.2, 6.3)
# ---------------------------------------------------------------------------

def test_planner_does_not_schedule_past_objects():
    """Objects whose gameplay window has passed are expired by the brain and
    must never be committed to the input queue or marked ACTION_SCHEDULED."""
    queue = CentralInputQueue()
    world_mgr = WorldStateManager()
    planner = RollingHorizonPlanner(queue=queue, world_manager=world_mgr)
    anchor_ns = clock.now_ns()

    circle = HitCircle(x=256.0, y=192.0, time_ms=100.0, hit_type=1)
    w = WorldObject("c_past", circle, ObjectLifecycleState.APPROACHING)
    world_mgr.add_or_update(w)

    # Plan one full second after the 100ms circle: long past its window.
    committed = planner.plan_cycle(
        current_time_ms=1000.0, timeline_anchor_ns=anchor_ns, mode=GameMode.STANDARD
    )

    assert len(committed) == 0
    assert len(queue) == 0
    assert world_mgr.get_object("c_past").lifecycle != ObjectLifecycleState.ACTION_SCHEDULED


# ---------------------------------------------------------------------------
# G-09: Cursor ownership & foreign move suppression (P0 7.3, 7.4, 7.6, P1 10)
# ---------------------------------------------------------------------------

def test_cursor_ownership_suppresses_foreign_moves_during_active_slider():
    """While a slider is actively tracking (holding key/down), foreign cursor
    movement intents (e.g. from an upcoming circle) are suppressed/cancelled."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)
    ctrl.set_backend(backend)

    # 1. Slider head key down establishes slider ownership
    slider_down = ActionIntent(
        action_id="act_s1_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=1000,
        deadline_ns=2000,
        target_id="slider_1",
        key="z",
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
        metadata={"is_slider": True},
    )
    ctrl.execute_action(slider_down)
    assert ctrl.cursor_owner_state == CursorOwnerState.SLIDER_FOLLOW
    assert ctrl.cursor_owner_target == "slider_1"

    # 2. Foreign circle move arrives while slider_1 is active
    foreign_move = ActionIntent(
        action_id="act_c2_move",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=1500,
        deadline_ns=2500,
        target_id="circle_2",
        x=500.0,
        y=500.0,
        cursor_owner=CursorOwnerState.CIRCLE_TARGET,
    )
    ctrl.execute_action(foreign_move)
    # Must be cancelled and NOT executed on backend
    assert foreign_move.status == InputStateStatus.CANCELLED
    assert ctrl.cursor_pos != (500.0, 500.0)

    # 3. Slider's own waypoint is allowed through
    slider_move = ActionIntent(
        action_id="act_s1_slide_1",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=1600,
        deadline_ns=2600,
        target_id="slider_1",
        x=300.0,
        y=300.0,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
    )
    ctrl.execute_action(slider_move)
    assert slider_move.status != InputStateStatus.CANCELLED
    assert ctrl.cursor_pos == (300.0, 300.0)

    # Cleanup
    ctrl.emergency_release_all()


def test_invalidated_target_waypoints_suppressed_by_controller():
    """Waypoints belonging to an invalidated/cancelled target are rejected
    even if popped from the queue."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)
    ctrl.set_backend(backend)

    # Invalidate target_A
    ctrl.invalidate_target("target_A")

    stale_move = ActionIntent(
        action_id="act_target_A_slide_5",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=1000,
        deadline_ns=2000,
        target_id="target_A",
        x=400.0,
        y=400.0,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
    )
    ctrl.execute_action(stale_move)
    assert stale_move.status == InputStateStatus.CANCELLED
    assert ctrl.cursor_pos != (400.0, 400.0)

    # Cleanup
    ctrl.emergency_release_all()


def test_cancel_active_slider_releases_ownership_and_invalidates():
    """cancel_active_slider releases the key, resets owner to IDLE, and
    marks target invalidated so lingering waypoints cannot execute."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)
    ctrl.set_backend(backend)

    slider_down = ActionIntent(
        action_id="act_s2_down",
        action_type=ActionType.KEY_DOWN,
        execute_at_ns=1000,
        deadline_ns=2000,
        target_id="slider_2",
        key="x",
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
        metadata={"is_slider": True},
    )
    ctrl.execute_action(slider_down)
    assert ctrl.cursor_owner_target == "slider_2"

    ctrl.cancel_active_slider("slider_2")
    assert ctrl.cursor_owner_state == CursorOwnerState.IDLE
    assert ctrl.cursor_owner_target is None

    # Subsequent move from cancelled slider_2 must be rejected
    lingering_move = ActionIntent(
        action_id="act_s2_slide_9",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=2000,
        deadline_ns=3000,
        target_id="slider_2",
        x=250.0,
        y=250.0,
        cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
    )
    ctrl.execute_action(lingering_move)
    assert lingering_move.status == InputStateStatus.CANCELLED

    # Cleanup
    ctrl.emergency_release_all()


def test_key_tap_and_key_down_do_not_alter_cursor_position():
    """KEY_TAP and KEY_DOWN are pure key inputs; they must never mutate cursor_pos."""
    backend = MockInputBackend()
    ctrl = CentralInputController(backend=backend)
    ctrl.set_backend(backend)

    # Commanded initial cursor position via MOUSE_MOVE
    init_move = ActionIntent(
        action_id="act_init",
        action_type=ActionType.MOUSE_MOVE,
        execute_at_ns=1000,
        deadline_ns=2000,
        target_id="circle_1",
        x=123.0,
        y=456.0,
    )
    ctrl.execute_action(init_move)
    assert ctrl.cursor_pos == (123.0, 456.0)

    # KEY_TAP carrying arbitrary (different) x, y in metadata/attributes
    tap = ActionIntent(
        action_id="act_tap",
        action_type=ActionType.KEY_TAP,
        execute_at_ns=1100,
        deadline_ns=2100,
        target_id="circle_1",
        x=999.0,
        y=888.0,
        key="z",
    )
    ctrl.execute_action(tap)
    # Cursor position must remain untouched by key actions
    assert ctrl.cursor_pos == (123.0, 456.0)

    # Cleanup
    ctrl.emergency_release_all()


# ---------------------------------------------------------------------------
# G-11: Beatmap title hint extraction handles parentheses & metadata safely
# ---------------------------------------------------------------------------

def test_beatmap_title_hint_handles_parentheses():
    """Song titles starting or containing parentheses must not be truncated to empty string."""
    import re

    test_cases = [
        (
            "Camellia - (The) Red * Room (TheShadowOfDark) [(osu!lazer) Debug]",
            "(The) Red * Room",
            "(osu!lazer) Debug",
        ),
        (
            "Artist - (Can You) Understand me? (Mapper) [Extra]",
            "(Can You) Understand me?",
            "Extra",
        ),
        (
            "Artist - Renai Circulation (TV Size) (Mapper) [Insane]",
            "Renai Circulation (TV Size)",
            "Insane",
        ),
        (
            "xi - Freedom Dive (Four Dimensions) [Another]",
            "Freedom Dive",
            "Another",
        ),
    ]

    for raw_working_map, expected_title, expected_diff in test_cases:
        m_diff = re.search(r"\[(.*?)\]", raw_working_map)
        assert m_diff is not None
        diff_name = m_diff.group(1).strip()
        assert diff_name == expected_diff

        base_part = raw_working_map.split("[")[0].strip()
        without_mapper = re.sub(r"\s*\([^)]*\)$", "", base_part).strip()
        title_hint = without_mapper.split(" - ", 1)[-1].strip() if " - " in without_mapper else without_mapper

        assert title_hint == expected_title
        assert len(title_hint) > 0


# ---------------------------------------------------------------------------
# G-12: Calling stop() from within worker thread does not raise join error
# ---------------------------------------------------------------------------

def test_app_stop_from_worker_thread_does_not_raise():
    """Calling stop() from the same thread as _session_thread must not raise cannot join current thread."""
    import threading
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend

    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())
    
    # Simulate being in worker thread
    dummy_thread = threading.current_thread()
    app._session_thread = dummy_thread
    app._is_running = True
    
    # Must not raise RuntimeError: cannot join current thread
    app.stop()
    assert not app._is_running
    assert app._session_thread is None


def test_win32_move_cursor_nan_guard_and_refocus_resilience():
    """Regression check for missing math import in move_cursor and deadman sync after window refocus."""
    from unittest.mock import patch
    from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend

    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    with patch.object(backend, "_get_osu_hwnd", return_value=12345), \
         patch.object(backend._user32, "IsWindow", return_value=True), \
         patch.object(backend._user32, "IsIconic", return_value=False), \
         patch.object(backend._user32, "GetForegroundWindow", return_value=12345), \
         patch.object(backend._user32, "SetCursorPos", return_value=1), \
         patch.object(backend._user32, "SendInput", return_value=1):
        assert backend.move_cursor(float("nan"), 500) is False
        backend._last_shiori_cursor_pos = (400, 400)
        backend._was_osu_active = False
        # Cursor moved 400px while another window held focus; refocusing must re-sync, NOT trigger 0.5s deadman blackout
        with patch.object(backend, "_get_current_cursor_pos", return_value=(900, 800)):
            assert backend.move_cursor(910, 805) is True
            assert backend._human_override_until_s == 0.0


def test_memory_mode_normalization_and_api_autotune(tmp_path):
    """Verify unknown-mode records normalize to standard, quit does not corrupt policy, API /memory works, and bubbly Live2D motions execute."""
    from fastapi.testclient import TestClient
    from src.gameplay.osu.api.server import app as fastapi_app
    from src.gameplay.osu.evaluation.game_memory import OsuGameMemory
    from src.gameplay.osu.timing.calibration import AdaptiveTimingCalibrator
    from src.gameplay.osu.integrations.vts_bridge import OsuVTSBridge
    from src.vtuber.motion_engine import IdleMotionEngine
    from src.vtuber.vts_client import VTSClient

    # 1. Calibrator holds converged offset inside <=1ms deadband without decaying to 0
    cal = AdaptiveTimingCalibrator(default_latency_ms=2.0)
    cal.set_adaptive_offset_ms(3.5)
    assert cal.consume_hit_errors([0.3, -0.2, 0.4]) == pytest.approx(3.5)

    # 2. Memory normalizes 'unknown' -> 'standard' and persists timing_errors
    mem = OsuGameMemory(memory_file_path=str(tmp_path / "mem.jsonl"))
    mem.__class__.profile_file = property(lambda self: tmp_path / "prof.json")
    mem.__class__.skill_file = property(lambda self: tmp_path / "skill.json")
    mem.log_session({
        "beatmap_id": 777001,
        "beatmap_title": "Bubbly Star",
        "mode": "unknown",
        "result": "pass",
        "accuracy": 99.2,
        "combo": 250,
        "max_combo": 250,
        "miss_count": 0,
        "hits": {"300": 250, "100": 2, "50": 0, "miss": 0},
        "avg_timing_error_ms": 4.0,
        "timing_errors": [3.8, 4.2, 4.0],
        "duration_s": 60.0,
    })
    prof = mem.get_beatmap_profile(777001, mode="standard")
    assert prof is not None and prof["mode"] == "standard"
    hist = mem.get_history(beatmap_id=777001)
    assert len(hist) == 1 and hist[0]["timing_errors"] == [3.8, 4.2, 4.0]

    # 3. REST API /api/osu/memory and autotune/live2d commands
    client = TestClient(fastapi_app)
    r_mem = client.get("/api/osu/memory")
    assert r_mem.status_code == 200 and "adaptive_policy" in r_mem.json()
    r_tune = client.post("/api/osu/command", json={"command": "autotune", "args": ""})
    assert r_tune.status_code == 200 and r_tune.json()["success"] is True
    r_l2d = client.post("/api/osu/command", json={"command": "live2d", "args": "star_wink"})
    assert r_l2d.status_code == 200 and r_l2d.json()["live2d"] == "star_wink"

    # 4. New bubbly Live2D spring actions & wink reflex
    engine = IdleMotionEngine(VTSClient())
    for act in ("happy_double_hop", "playful_wink_tilt", "rhythm_idol_bop", "excited_zoom_cheer", "cheeky_hip_sway", "starry_giggle_shake"):
        engine.trigger_action(act, allow_emotion_trigger=False)
        params = engine.compute_frame()
        assert any(p["id"] == "FaceAngleZ" for p in params)
    bridge = OsuVTSBridge()
    bridge.trigger_bop("cheer_hop", duration=1.0)
    assert bridge._active_bop_type == "cheer_hop"



# ---------------------------------------------------------------------------
# G-13: Song select preview time does not corrupt gameplay timeline anchor
# ---------------------------------------------------------------------------

def test_song_select_preview_does_not_corrupt_gameplay_anchor():
    """Tosu preview time during song select must never trigger fake drift or session rewind."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.integrations.tosu_client import tosu_client

    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())
    app._is_running = True

    # Simulate: Tosu is connected, but user is in SongSelect (state=5, is_playing=False)
    # reporting preview time 86,150ms!
    orig_conn = tosu_client._is_connected
    orig_play = tosu_client.is_playing
    orig_time = tosu_client.time_ms

    try:
        tosu_client._is_connected = True
        tosu_client.is_playing = False
        tosu_client.time_ms = 86150.0

        # Initial anchor
        init_anchor = 1_000_000_000
        app._timeline_anchor_ns = init_anchor
        app._last_tosu_time_ms = -1.0

        # Run one step
        res = app.step(simulated_time_ms=None)

        # Must NOT sync anchor to 86150ms or treat as active play!
        assert app._timeline_anchor_ns == init_anchor
        assert app._last_tosu_time_ms == -1.0
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_play
        tosu_client.time_ms = orig_time


def test_osu_v2_tosu_schema_and_post_slider_tight_gap_and_aim_off():
    """Verify osu! v2 upgrades: Tosu v2 payload parsing, auto_aim=False safety, and <12ms post-slider rescue."""
    from src.gameplay.osu.beatmap.models import CurveType, HitCircle, Slider
    from src.gameplay.osu.brains.standard_brain import StandardBrain
    from src.gameplay.osu.input.action_model import ActionType
    from src.gameplay.osu.integrations.tosu_client import TosuClient
    from src.gameplay.osu.timing.calibration import timing_calibrator
    from src.gameplay.osu.world.object_lifecycle import WorldObject

    # 1. Tosu v2 schema parsing (including negative lead-in time)
    tc = TosuClient()
    tc._parse_payload({
        "state": {"number": 2, "name": "play"},
        "play": {
            "combo": {"current": 128, "max": 128},
            "accuracy": 99.42,
            "hits": {"300": 120, "100": 2, "50": 0, "0": 0},
            "healthBar": {"normal": 96.5},
            "hitErrorArray": [-1.5, 0.8, 1.2],
            "unstableRate": 54.3,
            "failed": False,
            "mode": {"number": 0, "name": "osu"},
        },
        "beatmap": {
            "id": 424242,
            "set": 101010,
            "artist": "Camellia",
            "title": " crystallized ",
            "version": "Extreme",
            "time": {"live": -840.0},
        },
        "folders": {"songs": "E:/osu/Songs", "beatmap": "101010 Camellia"},
        "files": {"beatmap": "map.osu"},
    })
    assert tc.api_version == "v2"
    assert tc.is_playing is True
    assert tc.time_ms == -840.0
    assert tc.current_combo == 128
    assert tc.accuracy == 99.42
    assert tc.beatmap_id == 424242
    assert tc.hit_errors == [-1.5, 0.8, 1.2]

    # 2. StandardBrain with auto_aim_enabled=False must not raise UnboundLocalError
    brain = StandardBrain()
    brain.auto_aim_enabled = False
    c_manual = WorldObject(id="c_man", hit_object=HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1))
    man_intents = brain.plan_actions(900.0, [c_manual], timeline_anchor_ns=1_000_000_000)
    assert any(i.action_type == ActionType.KEY_TAP for i in man_intents)
    assert not any(i.action_type == ActionType.MOUSE_MOVE for i in man_intents)

    # 3. Post-slider ultra-tight gap (<12ms after visual tail) rescues aim without breaking slider safe tail
    brain2 = StandardBrain()
    brain2.auto_aim_enabled = True
    slider = Slider(
        x=100.0, y=100.0, time_ms=1000.0, hit_type=2,
        curve_type=CurveType.LINEAR, curve_points=[(200.0, 100.0)],
        repeat_count=1, pixel_length=100.0, duration_ms=400.0, end_time_ms=1400.0,
    )
    tight_circle = HitCircle(x=360.0, y=280.0, time_ms=1408.0, hit_type=1)  # 8ms after tail (<12ms safe_tail)
    anchor_ns = 1_000_000_000_000
    intents = brain2.plan_actions(
        900.0,
        [WorldObject(id="s_v2", hit_object=slider), WorldObject(id="c_v2", hit_object=tight_circle)],
        timeline_anchor_ns=anchor_ns,
    )
    c_moves = [i for i in intents if i.target_id == "c_v2" and i.action_type == ActionType.MOUSE_MOVE]
    c_taps = [i for i in intents if i.target_id == "c_v2" and i.action_type == ActionType.KEY_TAP]
    safe_tail_ns = timing_calibrator.calculate_dispatch_time_ns(1412.0, anchor_ns)
    assert len(c_moves) == 1
    assert c_moves[0].execute_at_ns >= safe_tail_ns
    assert c_taps[0].execute_at_ns >= c_moves[0].execute_at_ns + 2_000_000


def test_live_log_regressions_reconnect_focus_vts_logger_and_shiori_motion():
    """Verify fixes for live startup log: vts_bridge logger, auto-reconnect+focus without auto-stop spam, and Live2D body_y/singing choreography."""
    import time
    from unittest.mock import MagicMock, patch
    import src.gameplay.osu.integrations.vts_bridge as vb
    import src.gameplay.osu.perception.capture as osu_cap
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.integrations.discord_reporter import osu_discord_reporter
    from src.emotion.classifier import EmotionClassifier
    from src.vtuber.motion_engine import IdleMotionEngine
    from src.tts.providers.piper import PiperTTSProvider

    # 1. vts_bridge logger is defined (fixes NameError in _background_maintenance)
    assert hasattr(vb, "logger") and vb.logger is not None

    # 2. Missing osu! window enters standby without killing _is_running, then auto-focuses on reconnect; 0-hit stop skips Discord spam
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    tosu_client.stop()
    from src.gameplay.osu.evaluation.feedback import feedback_evaluator
    feedback_evaluator.reset()
    from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
    from src.gameplay.osu.input.controller import CentralInputController
    ctrl = CentralInputController(backend=Win32SendInputBackend(screen_width=1920, screen_height=1080))
    app = OsuGameplayApp(controller=ctrl)
    app._is_running = True
    app._last_missing_game_time = time.time() - 5.0
    with patch.object(osu_cap, "find_osu_window_hwnd", side_effect=[None, 99999]), \
         patch.object(osu_cap, "focus_osu_window", return_value=True) as mock_focus, \
         patch.object(ctrl, "emergency_release_all"), \
         patch("ctypes.windll.user32.IsWindow", return_value=True), \
         patch("ctypes.windll.user32.GetForegroundWindow", return_value=99999), \
         patch("ctypes.windll.user32.GetWindowTextLengthW", return_value=0), \
         patch.object(app, "_try_auto_discover_beatmap", return_value=False), \
         patch("time.sleep"):
        app._last_hwnd_scan_time = 0.0
        app.step(simulated_time_ms=None)
        assert app._is_running is True
        assert app._was_window_connected is False
        app._last_hwnd_scan_time = 0.0
        app.step(simulated_time_ms=None)
        assert app._was_window_connected is True
        assert mock_focus.called

    with patch.object(osu_discord_reporter, "send_game_summary") as mock_discord:
        app.stop()
        assert not mock_discord.called

    # 3. EmotionClassifier classifies 'cho vui nha ba yêu~ hihi ♡' as happy (not dull neutral)
    ec = EmotionClassifier()
    em = ec.classify("Con đang ngồi ngóng trông chờ ba nhắn tin nè ba ơi! Trò chuyện cho vui nha ba yêu~ hihi ♡")
    assert em.emotion == "happy"

    # 4. IdleMotionEngine passes vertical spring bounce into BodyAngleY and animates singing BPM choreography
    me = IdleMotionEngine(vts_client=MagicMock(), fps=60.0)
    me.set_speech_state(True, mouth_open=0.85, duration=8.0, is_singing=True, bpm=120.0)
    t0 = time.time()
    f1 = {p["id"]: p["value"] for p in me.compute_frame(t0 + 0.08)}
    f2 = {p["id"]: p["value"] for p in me.compute_frame(t0 + 0.33)}
    assert "MouthForm" in f1
    assert abs(f1["BodyAngleY"]) > 1.2 or abs(f2["BodyAngleY"]) > 1.2
    assert f1["FaceAngleZ"] != f2["FaceAngleZ"]

    # 5. PiperTTSProvider(use_cuda=True) auto-disables CUDA cleanly when ONNX lacks CUDAExecutionProvider (0s torch lag, 0 warnings)
    tts = PiperTTSProvider(use_cuda=True)
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        assert tts._use_cuda is False


def test_zero_input_hijack_when_not_playing_and_brain_context_overflow_recovery():
    """Ensure 0 SendInput calls when osu! is not playing and CommentBrain never fails on 'mấy h rùi con' or context overflow."""
    from unittest.mock import MagicMock, patch
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.beatmap.models import HitCircle
    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
    from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend
    from src.gameplay.osu.input.controller import CentralInputController
    from src.gameplay.osu.integrations.tosu_client import tosu_client
    from src.core.comment_brain import CommentBrain
    from src.runtime.llama_adapter import LlamaCppAdapter
    from src.runtime.base import ChatMessage, GenerationConfig

    # 1. Win32SendInputBackend.release_all() MUST NOT send blind MOUSEUP/KEYUP when Shiori pressed nothing
    backend = Win32SendInputBackend(screen_width=1920, screen_height=1080)
    with patch.object(backend._user32, "SendInput") as mock_sendinput:
        backend.release_all()
        assert mock_sendinput.call_count == 0

    # 2. Live OsuGameplayApp with loaded beatmap MUST dispatch 0 inputs when osu! is closed or not playing
    ctrl = CentralInputController(backend=backend)
    app = OsuGameplayApp(controller=ctrl)
    app._is_running = True
    app._timeline = HitObjectTimeline([HitCircle(x=256.0, y=192.0, time_ms=50.0, hit_type=1)])
    orig_conn, orig_play = tosu_client._is_connected, tosu_client.is_playing
    try:
        tosu_client._is_connected = False
        tosu_client.is_playing = False
        with patch("src.gameplay.osu.perception.capture.find_osu_window_hwnd", return_value=None), \
             patch.object(backend._user32, "SendInput") as mock_sendinput, \
             patch.object(backend._user32, "SetCursorPos") as mock_setpos:
            app._last_hwnd_scan_time = 0.0
            app.step(simulated_time_ms=None)
            assert mock_sendinput.call_count == 0
            assert mock_setpos.call_count == 0
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_play

    # 3. 'mấy h rùi con' resolves real clock time without falling back to neuron-stabilizing error
    mock_llm = MagicMock()
    mock_llm.chat.return_value = MagicMock(error="simulated error", text="")
    brain = CommentBrain(adapter=mock_llm)
    reply = brain.process_comment("mấy h rùi con", author="Roshi", is_roshi=True)
    assert "ổn định lại nơ-ron" not in reply
    assert "giờ" in reply and "phút" in reply

    # 4. LlamaCppAdapter auto-compacts and retries when create_chat_completion raises context window ValueError
    inst = LlamaCppAdapter.__new__(LlamaCppAdapter)
    inst.model_name = "test.gguf"
    inst.n_ctx = 2048
    inst._lock = __import__("threading").Lock()
    inst._llm = MagicMock()
    calls = []
    def _fake_create(**kw):
        calls.append(len(kw["messages"]))
        if len(calls) == 1:
            raise ValueError("Requested tokens (2358) exceed context window of 2048")
        return {"choices": [{"message": {"content": "Dạ bây giờ là 13:55 nè Ba yêu! ♡"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 400, "completion_tokens": 12}}
    inst._llm.create_chat_completion.side_effect = _fake_create
    res = inst.chat([
        ChatMessage(role="system", content="S" * 5000),
        ChatMessage(role="user", content="old 1"),
        ChatMessage(role="assistant", content="old 2"),
        ChatMessage(role="user", content="mấy h rùi con"),
    ], GenerationConfig(max_tokens=220))
    assert res.error is None
    assert "13:55" in res.text
    assert calls == [4, 2]


def test_playerloader_preview_and_zero_leadin_do_not_miss_first_note():
    """PlayerLoader preview audio + 0.0ms lead-in must not prematurely commit/fire Note 1."""
    from src.gameplay.osu.app import OsuGameplayApp
    from src.gameplay.osu.beatmap.models import HitCircle
    from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
    from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
    from src.gameplay.osu.integrations.tosu_client import tosu_client

    app = OsuGameplayApp()
    backend = MockInputBackend()
    app.controller.set_backend(backend)
    app._is_running = True
    app._timeline = HitObjectTimeline([
        HitCircle(x=396.0, y=50.0, time_ms=161.0, hit_type=1),
        HitCircle(x=400.0, y=74.0, time_ms=626.0, hit_type=1),
    ])

    orig_conn = tosu_client._is_connected
    orig_play = tosu_client.is_playing
    orig_time = tosu_client.time_ms

    try:
        # 1. SongSelect standby at preview 107,370ms
        tosu_client._is_connected = True
        tosu_client.is_playing = False
        tosu_client.time_ms = 107370.0
        app.step(simulated_time_ms=None)

        # 2. PlayerLoader: Tosu flips is_playing=True while preview audio continues at 107,500ms
        tosu_client.is_playing = True
        tosu_client.time_ms = 107500.0
        tosu_client.last_packet_ns = clock.now_ns()
        app.step(simulated_time_ms=None)
        assert app._tosu_anchor_synced is False
        assert len(app.planner._committed_actions) == 0

        # 3. SoloPlayer lead-in: Tosu clamps negative countdown (-1839ms..0ms) to 0.0ms
        tosu_client.time_ms = 0.0
        tosu_client.last_packet_ns = clock.now_ns()
        app.step(simulated_time_ms=None)
        assert app._tosu_anchor_synced is False
        assert "act_obj_161_396_50" not in app.planner._committed_actions
        assert app._pre_aimed_first_note is True

        # 4. Audio clock crosses 0ms -> Tosu reports 8.0ms; Note 1 (161ms) is now committed cleanly
        tosu_client.time_ms = 8.0
        tosu_client.last_packet_ns = clock.now_ns()
        app.step(simulated_time_ms=None)
        assert app._tosu_anchor_synced is True
        assert "act_obj_161_396_50" in app.planner._committed_actions
    finally:
        tosu_client._is_connected = orig_conn
        tosu_client.is_playing = orig_play
        tosu_client.time_ms = orig_time
