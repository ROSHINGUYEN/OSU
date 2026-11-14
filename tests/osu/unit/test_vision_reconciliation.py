"""
Unit Tests for Phase 3: Vision Detection, Object Tracking, Mode Detection, and Reconciliation.
"""

import numpy as np
import pytest

from src.gameplay.osu.beatmap.models import HitCircle, Slider
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, ModeDetectionStatus, state_store
from src.gameplay.osu.perception import (
    DetectedObject,
    ModeDetectionService,
    ObjectTracker,
    PlayfieldBounds,
    PlayfieldPreprocessor,
    TrackedObject,
    VisionDetector,
)
from src.gameplay.osu.world import (
    ObjectLifecycleState,
    ReconciliationMode,
    ReconciliationService,
    WorldObject,
    WorldStateManager,
)


def test_vision_detector_and_heuristics():
    bounds = PlayfieldBounds(x=100, y=50, width=800, height=600, scale=1.56)
    detector = VisionDetector(bounds)

    # Empty frame returns empty list
    empty_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    assert detector.detect_objects(empty_frame) == []


def test_object_tracker_association():
    tracker = ObjectTracker(max_distance=40.0)
    t0 = clock.now_ns()

    # Frame 1: One object at (100, 100)
    det1 = DetectedObject(x=100.0, y=100.0, radius=25.0, confidence=0.9, timestamp_ns=t0)
    tracks_f1 = tracker.update([det1], t0)
    assert len(tracks_f1) == 1
    t_id = tracks_f1[0].track_id

    # Frame 2: Object moves to (105, 105) after 16ms
    t1 = t0 + 16_000_000
    det2 = DetectedObject(x=105.0, y=105.0, radius=25.0, confidence=0.92, timestamp_ns=t1)
    tracks_f2 = tracker.update([det2], t1)

    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t_id
    assert tracks_f2[0].current_x == 105.0
    assert tracks_f2[0].velocity_x > 0.0


def test_mode_detection_service():
    service = ModeDetectionService()

    # Dedicated Standard Mode
    state = service.evaluate_mode(beatmap_mode=GameMode.STANDARD, vision_detected_mode=GameMode.STANDARD)
    assert state.mode == GameMode.STANDARD
    assert state.status == ModeDetectionStatus.MODE_CONFIRMED
    assert state.user_overridden is False

    # User override test
    service.set_user_override(GameMode.STANDARD)
    state_ov = service.evaluate_mode(beatmap_mode=GameMode.STANDARD)
    assert state_ov.user_overridden is True
    service.clear_user_override()


def test_world_state_manager_transitions():
    mgr = WorldStateManager()
    hit = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1)
    w_obj = WorldObject(id="obj_1", hit_object=hit, lifecycle=ObjectLifecycleState.PREDICTED)

    mgr.add_or_update(w_obj)
    assert mgr.get_object("obj_1") is not None
    assert mgr.get_object("obj_1").lifecycle == ObjectLifecycleState.PREDICTED

    # Advance lifecycle
    mgr.transition_state("obj_1", ObjectLifecycleState.ACTION_SCHEDULED)
    assert mgr.get_object("obj_1").lifecycle == ObjectLifecycleState.ACTION_SCHEDULED

    # Sync to CentralStateStore
    store_world = mgr.sync_to_central_store()
    assert "obj_1" in store_world.objects
    assert store_world.objects["obj_1"]["lifecycle"] == "action_scheduled"


def test_hybrid_reconciliation():
    mgr = WorldStateManager()
    reconciler = ReconciliationService(mode=ReconciliationMode.HYBRID, world_manager=mgr)
    prep = PlayfieldPreprocessor(screen_width=1920, screen_height=1080)

    # Expected beatmap object at 1000ms
    hit = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1)
    timeline = HitObjectTimeline([hit])

    # Convert hit coordinate to screen
    sx, sy = prep.osu_to_screen(256.0, 192.0)

    # Visual detection near expected screen coordinate
    visual_track = TrackedObject(
        track_id="trk_1",
        current_x=float(sx + 2),  # 2 pixels offset
        current_y=float(sy + 1),
        radius=30.0,
        approach_scale=1.1,
        obj_type="circle",
        confidence=0.95,
        first_seen_ns=0,
        last_seen_ns=0,
    )

    # Reconcile at t=950ms (within approach window)
    reconciled = reconciler.reconcile(
        current_time_ms=950.0,
        tracked_objects=[visual_track],
        preprocessor=prep,
        timeline=timeline,
    )

    assert len(reconciled) == 1
    w_obj = reconciled[0]
    # Matched with visual track -> CONFIRMED with high confidence
    assert w_obj.lifecycle == ObjectLifecycleState.CONFIRMED
    assert w_obj.confidence >= 0.95
    assert w_obj.screen_x == float(sx + 2)


def test_hybrid_reconciliation_with_window_offset():
    """Verify reconciliation correctly matches client visual tracks to screen beatmap with non-zero window offsets."""
    mgr = WorldStateManager()
    reconciler = ReconciliationService(mode=ReconciliationMode.HYBRID, world_manager=mgr)
    prep = PlayfieldPreprocessor(screen_width=1280, screen_height=720)
    prep.update_window_position(200, 150)

    # Place a beatmap circle at osu coordinate (256, 192)
    hit = HitCircle(x=256.0, y=192.0, time_ms=1000.0, hit_type=1)
    timeline = HitObjectTimeline([hit])

    # Preprocessor osu_to_screen gives desktop screen coordinate:
    exp_screen_x, exp_screen_y = prep.osu_to_screen(256.0, 192.0)
    assert exp_screen_x >= 200
    assert exp_screen_y >= 150

    # Camera captures client area, so tracked object is in client coordinates:
    client_x = float(exp_screen_x - 200 + 3)  # 3px deviation
    client_y = float(exp_screen_y - 150 - 2)  # 2px deviation

    visual_track = TrackedObject(
        track_id="trk_offset_1",
        current_x=client_x,
        current_y=client_y,
        radius=30.0,
        approach_scale=1.05,
        obj_type="circle",
        confidence=0.95,
        first_seen_ns=0,
        last_seen_ns=0,
    )

    reconciled = reconciler.reconcile(
        current_time_ms=950.0,
        tracked_objects=[visual_track],
        preprocessor=prep,
        timeline=timeline,
    )

    assert len(reconciled) == 1
    w_obj = reconciled[0]
    # Successfully matched despite 200px window offset!
    assert w_obj.lifecycle == ObjectLifecycleState.CONFIRMED
    assert w_obj.screen_x == pytest.approx(float(exp_screen_x + 3), abs=0.1)
    assert w_obj.screen_y == pytest.approx(float(exp_screen_y - 2), abs=0.1)


def test_world_state_prune_expired():
    """Verify WorldStateManager prunes dead / expired objects to prevent unbounded memory growth."""
    mgr = WorldStateManager()

    old_hit = HitCircle(x=100.0, y=100.0, time_ms=1000.0, hit_type=1)
    fresh_hit = HitCircle(x=200.0, y=200.0, time_ms=2000.0, hit_type=1)

    obj_old = WorldObject(id="old_1", hit_object=old_hit, lifecycle=ObjectLifecycleState.HIT_CONFIRMED)
    obj_fresh = WorldObject(id="fresh_1", hit_object=fresh_hit, lifecycle=ObjectLifecycleState.APPROACHING)

    mgr.add_or_update(obj_old)
    mgr.add_or_update(obj_fresh)

    assert len(mgr.active_objects) == 2

    # Prune at t=1800ms (old_1 is 800ms old, cutoff is 600ms)
    pruned_count = mgr.prune_expired(current_time_ms=1800.0, max_age_ms=600.0)
    assert pruned_count == 1
    assert "old_1" not in mgr.active_objects
    assert "fresh_1" in mgr.active_objects




