"""
Hybrid Beatmap and Vision Reconciliation Engine.
Fuses offline beatmap timelines with live visual tracking to produce trusted WorldState for osu! Standard.
"""

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Shiori.Osu.Reconciliation")

from src.gameplay.osu.beatmap.models import HitCircle, HitObject, Slider, Spinner
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.perception.tracker import TrackedObject
from src.gameplay.osu.world.object_lifecycle import (
    ObjectLifecycleState,
    WorldObject,
)
from src.gameplay.osu.world.world_state import (
    WorldStateManager,
    world_state_manager,
)


class ReconciliationMode(str, Enum):
    BEATMAP_FIRST = "beatmap_first"
    VISION_FIRST = "vision_first"
    HYBRID = "hybrid"


class ReconciliationService:
    """
    Fuses Beatmap timeline with live computer vision tracking for osu! Standard.
    Resolves discrepancies and advances object lifecycle states.
    """

    def __init__(
        self,
        mode: ReconciliationMode = ReconciliationMode.HYBRID,
        spatial_match_threshold: float = 60.0,
        approach_window_ms: float = 600.0,
        world_manager: Optional[WorldStateManager] = None,
    ):
        self.mode = mode
        self.spatial_match_threshold = spatial_match_threshold
        self.approach_window_ms = approach_window_ms
        self.world_manager = world_manager or world_state_manager

    def reconcile(
        self,
        current_time_ms: float,
        tracked_objects: List[TrackedObject],
        preprocessor: PlayfieldPreprocessor,
        timeline: Optional[HitObjectTimeline] = None,
        mode: Optional[Any] = None,
        is_taiko: Optional[bool] = None,
    ) -> List[WorldObject]:
        """
        Reconciles live tracked visual entities against expected beatmap objects.
        Returns the updated list of active WorldObjects.
        """
        reconciled_objects: List[WorldObject] = []

        # Prune expired / dead objects to keep working memory lean and fast (< 25 objects)
        if hasattr(self.world_manager, "prune_expired"):
            self.world_manager.prune_expired(current_time_ms)

        if timeline is not None and not timeline.is_empty:
            # Beatmap-First or Hybrid Mode
            lookahead_ms = current_time_ms + self.approach_window_ms
            lookbehind_ms = current_time_ms - 8000.0

            upcoming_hit_objects = timeline.get_objects_in_window(lookbehind_ms, lookahead_ms)
            matched_track_ids = set()

            for i, hit_obj in enumerate(upcoming_hit_objects):
                obj_id = f"obj_{int(hit_obj.time_ms)}_{int(hit_obj.x)}_{int(hit_obj.y)}"
                expected_sx, expected_sy = preprocessor.osu_to_screen(hit_obj.x, hit_obj.y)

                # Try to find a matching visual track
                best_track: Optional[TrackedObject] = None
                best_dist = float("inf")

                for track in tracked_objects:
                    if track.track_id in matched_track_ids:
                        continue
                    track_sx = preprocessor.window_offset_x + track.current_x
                    track_sy = preprocessor.window_offset_y + track.current_y
                    dist = math.hypot(track_sx - expected_sx, track_sy - expected_sy)
                    if dist < self.spatial_match_threshold and dist < best_dist:
                        best_dist = dist
                        best_track = track

                existing = self.world_manager.get_object(obj_id)

                if best_track is not None:
                    matched_track_ids.add(best_track.track_id)
                    # Hybrid Fusion: Object confirmed by both Beatmap & Vision (Desktop Screen Coordinates)
                    screen_x = preprocessor.window_offset_x + best_track.current_x
                    screen_y = preprocessor.window_offset_y + best_track.current_y
                    confidence = 0.98

                    lifecycle = ObjectLifecycleState.CONFIRMED
                    if existing and existing.lifecycle in [
                        ObjectLifecycleState.ACTION_SCHEDULED,
                        ObjectLifecycleState.ACTION_SENT,
                        ObjectLifecycleState.RESULT_PENDING,
                        ObjectLifecycleState.HIT_CONFIRMED,
                        ObjectLifecycleState.MISS_CONFIRMED,
                    ]:
                        lifecycle = existing.lifecycle

                    w_obj = WorldObject(
                        id=obj_id,
                        hit_object=hit_obj,
                        lifecycle=lifecycle,
                        screen_x=screen_x,
                        screen_y=screen_y,
                        confidence=confidence,
                    )
                else:
                    # Visible only in Beatmap (approaching or predicted)
                    dt = hit_obj.time_ms - current_time_ms
                    start_t = getattr(hit_obj, "time_ms", 0.0)
                    dur = getattr(hit_obj, "duration_ms", 0.0) or 0.0
                    raw_end = getattr(hit_obj, "end_time_ms", 0.0) or 0.0
                    obj_end = raw_end if raw_end > start_t else (start_t + dur)
                    obj_end_dt = obj_end - current_time_ms
                    if -50.0 <= dt <= self.approach_window_ms:
                        lifecycle = ObjectLifecycleState.APPROACHING
                        confidence = 0.88
                    elif dt > self.approach_window_ms:
                        lifecycle = ObjectLifecycleState.PREDICTED
                        confidence = 0.80
                    elif obj_end_dt >= -50.0:
                        lifecycle = ObjectLifecycleState.APPROACHING
                        confidence = 0.90
                    else:
                        lifecycle = ObjectLifecycleState.EXPIRED
                        confidence = 0.50

                    if existing and existing.lifecycle in [
                        ObjectLifecycleState.ACTION_SCHEDULED,
                        ObjectLifecycleState.ACTION_SENT,
                        ObjectLifecycleState.RESULT_PENDING,
                        ObjectLifecycleState.HIT_CONFIRMED,
                        ObjectLifecycleState.MISS_CONFIRMED,
                    ]:
                        lifecycle = existing.lifecycle

                    w_obj = WorldObject(
                        id=obj_id,
                        hit_object=hit_obj,
                        lifecycle=lifecycle,
                        screen_x=float(expected_sx),
                        screen_y=float(expected_sy),
                        confidence=confidence,
                    )

                self.world_manager.add_or_update(w_obj)
                reconciled_objects.append(w_obj)

            # Check for unmatched visual tracks (vision-first artifacts or unexpected notes)
            for track in tracked_objects:
                if track.track_id not in matched_track_ids:
                    if self.mode == ReconciliationMode.VISION_FIRST:
                        track_sx = preprocessor.window_offset_x + track.current_x
                        track_sy = preprocessor.window_offset_y + track.current_y
                        osu_x, osu_y = preprocessor.screen_to_osu(int(track_sx), int(track_sy))

                        scale = getattr(track, "approach_scale", 1.0)
                        from src.gameplay.osu.timing.hit_windows import HitWindowModel
                        from src.gameplay.osu.core.state_store import state_store
                        active_ar = getattr(state_store.beatmap, "ar", 9.0) or 9.0
                        time_to_hit_ms = HitWindowModel.approach_scale_to_time_ms(scale, ar=active_ar)
                        estimated_hit_time_ms = current_time_ms + time_to_hit_ms
                        synth_hit = HitCircle(x=osu_x, y=osu_y, time_ms=estimated_hit_time_ms, hit_type=1)

                        synth_id = f"vis_{track.track_id}"
                        existing = self.world_manager.get_object(synth_id)
                        lifecycle = ObjectLifecycleState.CONFIRMED if getattr(track, "approach_scale", 1.0) <= 1.05 else ObjectLifecycleState.APPROACHING
                        if existing and existing.lifecycle in [
                            ObjectLifecycleState.ACTION_SCHEDULED,
                            ObjectLifecycleState.ACTION_SENT,
                            ObjectLifecycleState.HIT_CONFIRMED,
                            ObjectLifecycleState.MISS_CONFIRMED,
                        ]:
                            lifecycle = existing.lifecycle

                        w_obj = WorldObject(
                            id=synth_id,
                            hit_object=synth_hit,
                            lifecycle=lifecycle,
                            screen_x=track_sx,
                            screen_y=track_sy,
                            confidence=0.95,
                        )
                        self.world_manager.add_or_update(w_obj)
                        reconciled_objects.append(w_obj)

        elif self.mode == ReconciliationMode.VISION_FIRST or timeline is None:
            # Pure Vision-first for Standard mode
            from src.gameplay.osu.core.state_store import state_store
            for track in tracked_objects:
                track_sx = preprocessor.window_offset_x + track.current_x
                track_sy = preprocessor.window_offset_y + track.current_y
                osu_x, osu_y = preprocessor.screen_to_osu(int(track_sx), int(track_sy))

                scale = getattr(track, "approach_scale", 1.0)
                from src.gameplay.osu.timing.hit_windows import HitWindowModel
                active_ar = getattr(state_store.beatmap, "ar", 9.0) or 9.0
                time_to_hit_ms = HitWindowModel.approach_scale_to_time_ms(scale, ar=active_ar)
                estimated_hit_time_ms = current_time_ms + time_to_hit_ms
                synth_hit = HitCircle(x=osu_x, y=osu_y, time_ms=estimated_hit_time_ms, hit_type=1)

                synth_id = f"vis_{track.track_id}"
                existing = self.world_manager.get_object(synth_id)
                lifecycle = ObjectLifecycleState.CONFIRMED if getattr(track, "approach_scale", 1.0) <= 1.05 else ObjectLifecycleState.APPROACHING
                if existing and existing.lifecycle in [
                    ObjectLifecycleState.ACTION_SCHEDULED,
                    ObjectLifecycleState.ACTION_SENT,
                    ObjectLifecycleState.HIT_CONFIRMED,
                    ObjectLifecycleState.MISS_CONFIRMED,
                ]:
                    lifecycle = existing.lifecycle

                w_obj = WorldObject(
                    id=synth_id,
                    hit_object=synth_hit,
                    lifecycle=lifecycle,
                    screen_x=track_sx,
                    screen_y=track_sy,
                    confidence=0.95,
                )
                self.world_manager.add_or_update(w_obj)
                reconciled_objects.append(w_obj)

        # Sync to central store
        self.world_manager.sync_to_central_store()
        return reconciled_objects


reconciliation_service = ReconciliationService()
