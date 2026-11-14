"""
osu!standard Gameplay Brain.
Handles cursor aiming trajectory generation (minimum-jerk), deterministic key alternation (z/x),
slider path following, spinner rotation, and slider preservation.
"""

import math
import hashlib
from typing import Dict, List, Optional, Tuple

from src.gameplay.osu.beatmap.models import HitCircle, HitObject, Slider, Spinner
from src.gameplay.osu.brains.base import BaseGameplayBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.gpu_engine import gpu_engine
from src.gameplay.osu.core.state_store import GameMode
from src.gameplay.osu.input.action_model import ActionIntent, ActionType, CursorOwnerState
from src.gameplay.osu.math.slider_curves import calculate_slider_path
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.timing.calibration import timing_calibrator
from src.gameplay.osu.timing.hit_windows import HitWindowModel
from src.gameplay.osu.safety.humanizer import humanizer
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject


class StandardBrain(BaseGameplayBrain):
    """Cursor aiming trajectory generator and click scheduler for osu!standard."""

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        od: float = 8.0,
        cs: float = 4.0,
        preprocessor: Optional[PlayfieldPreprocessor] = None,
        smoothing_steps: int = 4,
    ):
        super().__init__(GameMode.STANDARD)
        self.keys = keys or ["z", "x"]
        self.od = od
        self.cs = float(cs)
        self.preprocessor = preprocessor or PlayfieldPreprocessor()
        self.smoothing_steps = max(2, smoothing_steps)
        self.auto_aim_enabled: bool = True
        self.humanize_anti_detect: bool = False  # Optional humanization variance to avoid robotic replay signatures

        self._cursor_x: float = 256.0
        self._cursor_y: float = 192.0
        self._last_assigned_key: str = "x"
        self._last_assigned_time_ms: float = -1000.0
        self._last_assigned_end_time_ms: float = -1000.0
        self._last_assigned_is_slider: bool = False
        self._last_key_idx = 0
        self._assigned_keys: Dict[str, str] = {}
        self._active_slider_key: Optional[str] = None
        self._active_slider_end_ms: float = 0.0
        self._active_sliders: Dict[str, Tuple[float, float, str]] = {}
        self._last_completed_time_ms: float = 0.0
        self.cursor_owner_state: CursorOwnerState = CursorOwnerState.IDLE

    def reset(self) -> None:
        self._cursor_x = 256.0
        self._cursor_y = 192.0
        self._last_assigned_key = "x"
        self._last_assigned_time_ms = -1000.0
        self._last_assigned_end_time_ms = -1000.0
        self._last_assigned_is_slider = False
        self._last_key_idx = 0
        self._assigned_keys.clear()
        self._active_slider_key = None
        self._active_slider_end_ms = 0.0
        self._active_sliders.clear()
        self._last_completed_time_ms = 0.0
        self.cursor_owner_state = CursorOwnerState.IDLE

    @staticmethod
    def _minimum_jerk_step(u: float) -> float:
        """Minimum-jerk polynomial trajectory: s(u) = 10u^3 - 15u^4 + 6u^5."""
        u = max(0.0, min(1.0, u))
        return (10.0 * (u ** 3)) - (15.0 * (u ** 4)) + (6.0 * (u ** 5))

    @staticmethod
    def _interpolate_slider_path(
        pts: List[Tuple[float, float]],
        progress: float,
        repeat_count: int = 1,
        cum_lens: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """
        Calculates slider ball position (x, y) in osu! coordinates for progress in [0..1].
        Supports repeat counts (ping-pong reversing direction on odd repeats).
        Uses precomputed cumulative arc lengths when provided for O(log N) lookup.
        """
        if not pts:
            return 256.0, 192.0
        if len(pts) == 1:
            return pts[0]

        span_count = max(1, repeat_count)
        if progress >= 1.0:
            frac = 0.0 if (span_count % 2 == 0) else 1.0
        else:
            total_p = progress * span_count
            span = int(total_p)
            frac = total_p - span
            if span % 2 == 1:
                frac = 1.0 - frac
        frac = max(0.0, min(1.0, frac))

        if cum_lens is None:
            cum_lens = [0.0]
            for i in range(len(pts) - 1):
                d = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
                cum_lens.append(cum_lens[-1] + d)

        total_len = cum_lens[-1]
        if total_len <= 0.0:
            return pts[0]

        target_dist = frac * total_len
        import bisect
        idx = bisect.bisect_right(cum_lens, target_dist) - 1
        idx = max(0, min(len(pts) - 2, idx))
        seg_len = cum_lens[idx + 1] - cum_lens[idx]
        t = (target_dist - cum_lens[idx]) / max(1e-6, seg_len)
        t = max(0.0, min(1.0, t))
        x = pts[idx][0] + (pts[idx+1][0] - pts[idx][0]) * t
        y = pts[idx][1] + (pts[idx+1][1] - pts[idx][1]) * t
        return x, y

    def plan_actions(
        self,
        current_time_ms: float,
        objects: List[WorldObject],
        timeline_anchor_ns: int,
        observed_cursor_pos: Optional[Tuple[float, float]] = None,
    ) -> List[ActionIntent]:
        intents: List[ActionIntent] = []
        from src.gameplay.osu.planning.adaptive_policy import adaptive_policy
        w300, w100, w50 = HitWindowModel.standard(self.od)
        # Use full w50 hit window (~115ms) so micro-delays don't cause premature stale dropping
        tolerance_ns = int(w50 * 1_000_000)

        # Incorporate real observed/commanded cursor position if provided and not in active slider tracking
        if observed_cursor_pos is not None:
            obs_sx, obs_sy = observed_cursor_pos
            is_actively_tracking_slider = any(
                s[0] <= current_time_ms <= s[1] for s in self._active_sliders.values()
            )
            if obs_sx > 0.0 and obs_sy > 0.0 and not is_actively_tracking_slider:
                real_ox, real_oy = self.preprocessor.screen_to_osu(int(obs_sx), int(obs_sy))
                if 0.0 <= real_ox <= 512.0 and 0.0 <= real_oy <= 384.0:
                    self._cursor_x = real_ox
                    self._cursor_y = real_oy

        # Prune assigned keys cache if map is very long (> 2000 objects)
        if len(self._assigned_keys) > 2000:
            self._assigned_keys.clear()

        # Prune expired sliders from active tracker
        self._active_sliders = {
            sid: val for sid, val in self._active_sliders.items()
            if current_time_ms <= (val[1] + 50.0)
        }
        if self._active_slider_key and current_time_ms > (self._active_slider_end_ms + 50.0):
            self._active_slider_key = None

        # Sort objects by hit time to ensure deterministic sequential assignment
        sorted_objects = sorted(objects, key=lambda o: getattr(o.hit_object, "time_ms", 0.0))

        # Simulation cursor & key state across this horizon iteration
        sim_cursor_x = self._cursor_x
        sim_cursor_y = self._cursor_y
        sim_last_key = self._last_assigned_key
        sim_last_time_ms = self._last_assigned_time_ms
        sim_last_end_time_ms = self._last_completed_time_ms
        sim_last_is_slider = getattr(self, "_last_assigned_is_slider", False)
        if sim_last_end_time_ms <= 0.0 or (current_time_ms - sim_last_end_time_ms) > 5000.0 or sim_last_end_time_ms > (current_time_ms + 15000.0):
            sim_last_end_time_ms = current_time_ms

        for w_obj in sorted_objects:
            hit = w_obj.hit_object
            target_x_osu = max(0.0, min(512.0, float(hit.x)))
            target_y_osu = max(0.0, min(384.0, float(hit.y)))
            target_t_ms = hit.time_ms

            # Compute object end position and duration
            if isinstance(hit, Slider):
                if getattr(hit, "_cached_path", None) is None:
                    hit._cached_path = calculate_slider_path(hit)
                    cum_lens = [0.0]
                    for i in range(len(hit._cached_path) - 1):
                        d = math.hypot(hit._cached_path[i+1][0] - hit._cached_path[i][0], hit._cached_path[i+1][1] - hit._cached_path[i][1])
                        cum_lens.append(cum_lens[-1] + d)
                    hit._cached_cum_lens = cum_lens
                all_pts = hit._cached_path
                cum_lens = hit._cached_cum_lens
                repeat_count = getattr(hit, "repeat_count", 1)
                obj_end_x, obj_end_y = self._interpolate_slider_path(all_pts, 1.0, repeat_count, cum_lens=cum_lens)
                duration_ms = float(hit.duration_ms) if getattr(hit, "duration_ms", 0.0) > 0.0 else 50.0
            elif isinstance(hit, Spinner):
                obj_end_x, obj_end_y = 256.0, 192.0
                duration_ms = max(100.0, getattr(hit, "duration_ms", 500.0))
            else:
                obj_end_x, obj_end_y = target_x_osu, target_y_osu
                duration_ms = 0.0

            # 0. Expiration Filter: Drop unhittable past objects.
            # For sliders/spinners, use end time; for circles, use hit time.
            obj_end_ms = target_t_ms + duration_ms
            if obj_end_ms < (current_time_ms - 40.0):
                w_obj.transition_to(ObjectLifecycleState.EXPIRED)
                continue

            if w_obj.lifecycle in [
                ObjectLifecycleState.ACTION_SCHEDULED,
                ObjectLifecycleState.ACTION_SENT,
                ObjectLifecycleState.HIT_CONFIRMED,
                ObjectLifecycleState.MISS_CONFIRMED,
                ObjectLifecycleState.EXPIRED,
            ]:
                # Object was already scheduled in a previous planning cycle:
                # Advance simulated cursor position and active state to where this object finishes
                sim_cursor_x = obj_end_x
                sim_cursor_y = obj_end_y
                if target_t_ms + duration_ms <= current_time_ms:
                    self._cursor_x = obj_end_x
                    self._cursor_y = obj_end_y
                    self._last_completed_time_ms = max(self._last_completed_time_ms, target_t_ms + duration_ms)
                is_sl = isinstance(hit, Slider)
                sim_last_is_slider = is_sl
                sim_last_end_time_ms = max(sim_last_end_time_ms, target_t_ms + duration_ms + (12.0 if is_sl else 0.0))
                val = self._assigned_keys.get(w_obj.id)
                if val:
                    s_key = val[0] if isinstance(val, tuple) else val
                    sim_last_key = s_key
                    sim_last_time_ms = target_t_ms
                    if target_t_ms <= current_time_ms:
                        self._last_assigned_key = s_key
                        self._last_assigned_time_ms = target_t_ms
                        self._last_assigned_is_slider = is_sl
                        self._last_assigned_end_time_ms = sim_last_end_time_ms
                if isinstance(hit, Slider):
                    s_key = val[0] if (val and isinstance(val, tuple)) else (val or "z")
                    end_t = target_t_ms + duration_ms
                    safe_end = end_t + 12.0
                    self._active_sliders[w_obj.id] = (target_t_ms, safe_end, s_key, end_t)
                    self._active_slider_end_ms = max(self._active_slider_end_ms, safe_end)
                continue

            if sim_last_time_ms > target_t_ms:
                sim_last_time_ms = -1000.0

            if getattr(w_obj, "screen_x", 0.0) > 0.0 and getattr(w_obj, "screen_y", 0.0) > 0.0:
                target_sx = int(w_obj.screen_x)
                target_sy = int(w_obj.screen_y)
            else:
                target_sx, target_sy = self.preprocessor.osu_to_screen(target_x_osu, target_y_osu)
            curr_sx, curr_sy = self.preprocessor.osu_to_screen(sim_cursor_x, sim_cursor_y)

            dispatch_ns = timing_calibrator.calculate_dispatch_time_ns(target_t_ms, timeline_anchor_ns)
            effective_dispatch_ns = dispatch_ns
            if humanizer.is_enabled or self.humanize_anti_detect:
                # Anti-Ban / Stealth Biometric Humanization: Gaussian jitter & spatial dispersion
                t_jitter_ns = humanizer.calculate_hit_timing_jitter_ns(w_obj.id, self.od)
                effective_dispatch_ns += t_jitter_ns
                circle_radius_px = getattr(self.preprocessor, "circle_radius_px", 32.0) or 32.0
                target_sx, target_sy = humanizer.humanize_click_position(target_sx, target_sy, circle_radius_px, w_obj.id)
            deadline_ns = dispatch_ns + tolerance_ns

            allow_mouse = getattr(self, "auto_aim_enabled", True)
            earliest_move_ms = max(current_time_ms, sim_last_end_time_ms)
            post_slider_rescued = False

            # 1. Cursor Aiming: Smooth, snappy trajectory arriving 12-18ms BEFORE hit window
            aim_lead_time = float(adaptive_policy.get_param("aim_lead_time_ms", 16.0))
            slider_step_ms = float(adaptive_policy.get_param("slider_step_ms", 22.0))
            dyn_smooth_steps = self.smoothing_steps

            if allow_mouse:
                active_slider_body_end_ms = -1000.0
                for _sid, s_data in self._active_sliders.items():
                    s_start = s_data[0]
                    s_safe_end = s_data[1]
                    s_cursor_end = s_data[3] if len(s_data) > 3 else s_data[1]
                    if target_t_ms > s_start:
                        # Ensure cursor stays locked on slider until at least s_cursor_end (slider end time)
                        leave_t = s_cursor_end
                        earliest_move_ms = max(earliest_move_ms, leave_t)
                        active_slider_body_end_ms = max(active_slider_body_end_ms, s_cursor_end)
                    if s_start <= current_time_ms < s_cursor_end:
                        earliest_move_ms = max(earliest_move_ms, s_cursor_end)
                        active_slider_body_end_ms = max(active_slider_body_end_ms, s_cursor_end)

                dist_px = math.hypot(target_sx - curr_sx, target_sy - curr_sy)
                cs_val = float(getattr(hit, "cs", 0.0) or getattr(self, "cs", 4.0) or 4.0)
                lead_min = 16.0 if cs_val >= 5.0 else 12.0
                lead_max = 30.0 if cs_val >= 5.0 else 22.0
                lead_time_ms = max(lead_min, min(lead_max, aim_lead_time + (6.0 if cs_val >= 5.0 else 0.0)))
                arrive_ms = target_t_ms - lead_time_ms
                available_time_ms = target_t_ms - earliest_move_ms

                # SLIDER CURSOR LOCK: If target hit time is while a slider still owns the cursor,
                # do NOT spawn conflicting cursor movements that pull the mouse away!
                if target_t_ms < earliest_move_ms:
                    # osu! v2 post-slider tight gap rescue (0 <= gap < 12ms after visual slider tail):
                    # Keep slider safe tail intact (>= earliest_move_ns) and snap + tap inside Note B's w300 window!
                    if target_t_ms >= active_slider_body_end_ms and (earliest_move_ms - target_t_ms) <= 20.0:
                        earliest_move_ns = timing_calibrator.calculate_dispatch_time_ns(earliest_move_ms, timeline_anchor_ns)
                        move_ns = min(earliest_move_ns, deadline_ns)
                        intents.append(
                            ActionIntent(
                                action_id=f"act_{w_obj.id}_move_1",
                                action_type=ActionType.MOUSE_MOVE,
                                execute_at_ns=move_ns,
                                deadline_ns=deadline_ns,
                                mode=GameMode.STANDARD,
                                target_id=w_obj.id,
                                x=float(target_sx),
                                y=float(target_sy),
                                created_at_ns=clock.now_ns(),
                                target_time_ms=target_t_ms,
                                priority=2,
                                cursor_owner=CursorOwnerState.CIRCLE_TARGET,
                            )
                        )
                        effective_dispatch_ns = min(deadline_ns, max(dispatch_ns, move_ns + 2_000_000))
                        post_slider_rescued = True
                elif available_time_ms <= 15.0 or earliest_move_ms >= arrive_ms:
                    # Snappy single-step arrival to guarantee cursor is on target before click
                    # NEVER pull cursor away before earliest_move_ms (active slider tail or dwell window)!
                    arrival_lead = max(6.0, min(lead_time_ms, available_time_ms * 0.7)) if available_time_ms > 8.0 else 2.0
                    target_move_ms = max(earliest_move_ms, target_t_ms - arrival_lead)
                    start_move_ns = timing_calibrator.calculate_dispatch_time_ns(target_move_ms, timeline_anchor_ns)
                    earliest_move_ns = timing_calibrator.calculate_dispatch_time_ns(earliest_move_ms, timeline_anchor_ns)
                    move_ns = max(earliest_move_ns, start_move_ns)
                    intents.append(
                        ActionIntent(
                            action_id=f"act_{w_obj.id}_move_1",
                            action_type=ActionType.MOUSE_MOVE,
                            execute_at_ns=min(move_ns, deadline_ns),
                            deadline_ns=deadline_ns,
                            mode=GameMode.STANDARD,
                            target_id=w_obj.id,
                            x=float(target_sx),
                            y=float(target_sy),
                            created_at_ns=clock.now_ns(),
                            target_time_ms=target_t_ms,
                            priority=2,
                            cursor_owner=CursorOwnerState.CIRCLE_TARGET,
                        )
                    )
                else:
                    required_vel = dist_px / max(1.0, available_time_ms)
                    # Adaptive landing lead: for tight gaps, scale lead proportionally so motion duration is healthy
                    effective_lead_ms = max(8.0, min(lead_time_ms, available_time_ms * 0.35))

                    if available_time_ms <= 220.0:
                        # Rapid jumps, 1/2 beats, and high-tempo transitions:
                        # Depart promptly from earliest_move_ms to maximize travel time and landing stability!
                        start_move_ms = earliest_move_ms
                        motion_duration_ms = max(4.0, (target_t_ms - effective_lead_ms) - earliest_move_ms)
                        if available_time_ms < 50.0:
                            steps = 1
                        elif available_time_ms < 85.0:
                            steps = 1 if (dist_px < 35.0 or available_time_ms < 65.0) else 2
                        elif available_time_ms < 135.0:
                            steps = 2 if (available_time_ms < 110.0 and dist_px < 150.0) else min(dyn_smooth_steps, 3)
                        else:
                            steps = dyn_smooth_steps
                    else:
                        # Generous / Slower transitions (> 220ms, e.g. 1/1 beats or multi-beat breaks):
                        # Arrive generously 35ms - 65ms before hit so cursor rests rock-solid at center
                        effective_lead_ms = max(35.0, min(65.0, available_time_ms * 0.12 if cs_val >= 5.0 else available_time_ms * 0.08))
                        travel_window_ms = min(220.0, max(30.0, available_time_ms - effective_lead_ms))
                        motion_duration_ms = travel_window_ms
                        start_move_ms = max(earliest_move_ms, (target_t_ms - effective_lead_ms) - motion_duration_ms)
                        steps = dyn_smooth_steps

                    start_move_ns = timing_calibrator.calculate_dispatch_time_ns(start_move_ms, timeline_anchor_ns)
                    earliest_move_ns = timing_calibrator.calculate_dispatch_time_ns(earliest_move_ms, timeline_anchor_ns)
                    step_dt_ns = int((motion_duration_ms / steps) * 1_000_000)

                    # Final aim arrival: ensure the final step lands on target center BEFORE key dispatch
                    final_lead_ns = int(effective_lead_ms * 1_000_000)
                    aim_arrival_ns = dispatch_ns - final_lead_ns

                    traj = gpu_engine.compute_minimum_jerk_trajectory(
                        (curr_sx, curr_sy),
                        (target_sx, target_sy),
                        steps=steps,
                    )
                    for step, (interp_x, interp_y, _u) in enumerate(traj, start=1):
                        target_step_ns = start_move_ns + (step * step_dt_ns)
                        if step == steps:
                            move_ns = max(earliest_move_ns, min(target_step_ns, aim_arrival_ns))
                        else:
                            move_ns = max(earliest_move_ns, min(target_step_ns, aim_arrival_ns - 4_000_000))
                        intents.append(
                            ActionIntent(
                                action_id=f"act_{w_obj.id}_move_{step}",
                                action_type=ActionType.MOUSE_MOVE,
                                execute_at_ns=min(move_ns, deadline_ns),
                                deadline_ns=deadline_ns,
                                mode=GameMode.STANDARD,
                                target_id=w_obj.id,
                                x=interp_x,
                                y=interp_y,
                                created_at_ns=clock.now_ns(),
                                target_time_ms=target_t_ms,
                                priority=2,
                                cursor_owner=CursorOwnerState.MOVING_TO_TARGET,
                            )
                        )

            # 2. Key Assignment (z/x) with strict Slider Protection & Single-Tap + Stream Alternation
            delta_t = target_t_ms - sim_last_time_ms
            gap_from_last_end = target_t_ms - sim_last_end_time_ms
            if w_obj.id in self._assigned_keys:
                val = self._assigned_keys[w_obj.id]
                key = val[0] if isinstance(val, tuple) else val
                tap_hold = val[1] if isinstance(val, tuple) and len(val) > 1 else 25.0
                is_slider_locked = val[2] if isinstance(val, tuple) and len(val) > 2 else False
            else:
                locked_keys = {
                    s_data[2] for _sid, s_data in self._active_sliders.items()
                    if s_data[0] <= target_t_ms <= (s_data[1] + 35.0)
                }
                is_slider_locked = bool(locked_keys)

                # Alternation criteria dynamically fine-tuned by AdaptivePlanningPolicy & Diagnostics:
                alt_thresh = float(adaptive_policy.get_param("alternation_threshold_ms", 200.0))
                post_slider_thresh = float(adaptive_policy.get_param("post_slider_gap_ms", 260.0))
                is_rapid_burst = (sim_last_time_ms > 0 and delta_t <= alt_thresh)
                is_post_slider_transition = (sim_last_is_slider and (gap_from_last_end <= post_slider_thresh or delta_t <= (post_slider_thresh + 100.0)))
                should_alternate = is_rapid_burst or is_post_slider_transition

                if should_alternate:
                    # Stream / burst / fast 1/2-beat / post-slider notes: alternate strictly between keys
                    alt_key = self.keys[1] if sim_last_key == self.keys[0] else self.keys[0]
                    if alt_key in locked_keys:
                        # Fallback if alt_key is held by an active slider
                        available = [k for k in self.keys if k not in locked_keys]
                        key = available[0] if available else alt_key
                    else:
                        key = alt_key
                elif locked_keys:
                    # Key is locked by active slider: pick the unlocked key so active slider is never broken
                    available = [k for k in self.keys if k not in locked_keys]
                    key = available[0] if available else self.keys[0]
                else:
                    # Standard Single-Tap:
                    # For jumps, slow 1/1 beats with long breaks (delta_t > 220ms and gap > 260ms):
                    # Always use primary key (keys[0] = 'z')!
                    key = self.keys[0]

                # Ensure tap_hold is long enough (20ms - 35ms) for Windows SendInput & osu!lazer frame polling,
                # while leaving sufficient release window before the same key is hit again.
                if should_alternate:
                    effective_dt = max(45.0, delta_t) if delta_t > 0 else 50.0
                    tap_hold = min(28.0, max(20.0, effective_dt * 0.45))
                else:
                    tap_hold = min(35.0, max(22.0, max(0.0, delta_t) * 0.35))
                self._assigned_keys[w_obj.id] = (key, tap_hold, is_slider_locked)

            sim_last_key = key
            sim_last_time_ms = target_t_ms

            if isinstance(hit, Slider):
                end_t_ms = target_t_ms + duration_ms
                # Pro-mechanic: Hold slider key for +12ms past tail to guarantee game engine registers tail tick without premature drop
                safe_tail_ms = end_t_ms + 12.0
                end_dispatch_ns = timing_calibrator.calculate_dispatch_time_ns(safe_tail_ms, timeline_anchor_ns)

                self._active_slider_key = key
                self._active_slider_end_ms = safe_tail_ms
                self._active_sliders[w_obj.id] = (target_t_ms, safe_tail_ms, key, end_t_ms)

                # Slider Start: Key Down (Keyboard Only)
                intents.append(
                    ActionIntent(
                        action_id=f"act_{w_obj.id}_down",
                        action_type=ActionType.KEY_DOWN,
                        execute_at_ns=effective_dispatch_ns,
                        deadline_ns=deadline_ns,
                        mode=GameMode.STANDARD,
                        target_id=w_obj.id,
                        x=float(target_sx),
                        y=float(target_sy),
                        key=key,
                        button=None,
                        created_at_ns=clock.now_ns(),
                        target_time_ms=target_t_ms,
                        priority=3,
                        metadata={
                            "is_slider": True,
                            "duration_ms": duration_ms,
                            "repeat_count": repeat_count,
                        },
                    )
                )

                if allow_mouse:
                    span_count = max(1, repeat_count)
                    span_duration_ms = duration_ms / float(span_count)
                    slider_len = cum_lens[-1] if (cum_lens and len(cum_lens) > 0) else 100.0
                    slider_vel = slider_len / max(1.0, span_duration_ms)
                    # Adaptive slider step: for fast sliders (high velocity px/ms), use tighter step (8-12ms)
                    if slider_vel > 2.2:
                        effective_step_ms = min(slider_step_ms, 8.0)
                    elif slider_vel > 1.6:
                        effective_step_ms = min(slider_step_ms, 10.0)
                    elif slider_vel > 1.2:
                        effective_step_ms = min(slider_step_ms, 12.0)
                    else:
                        effective_step_ms = slider_step_ms
                    step_idx = 0
                    for k in range(span_count):
                        # Short buzz sliders (len < 35px or span_dur < 55ms): clamp to 1-2 steps per span
                        if slider_len < 35.0 or span_duration_ms < 55.0:
                            span_steps = max(1, min(2, int(span_duration_ms / 20.0)))
                        else:
                            span_steps = max(4, int(span_duration_ms / effective_step_ms))
                        for i in range(1, span_steps + 1):
                            step_idx += 1
                            u = i / float(span_steps)
                            p = (k + u) / float(span_count)
                            ball_x_osu, ball_y_osu = self._interpolate_slider_path(
                                all_pts, p, repeat_count, cum_lens=cum_lens
                            )
                            ball_sx, ball_sy = self.preprocessor.osu_to_screen(ball_x_osu, ball_y_osu)
                            s_move_ms = target_t_ms + (p * duration_ms)
                            s_move_ns = timing_calibrator.calculate_dispatch_time_ns(s_move_ms, timeline_anchor_ns)

                            intents.append(
                                ActionIntent(
                                    action_id=f"act_{w_obj.id}_slide_{step_idx}",
                                    action_type=ActionType.MOUSE_MOVE,
                                    execute_at_ns=s_move_ns,
                                    deadline_ns=s_move_ns + 50_000_000,
                                    mode=GameMode.STANDARD,
                                    target_id=w_obj.id,
                                    x=float(ball_sx),
                                    y=float(ball_sy),
                                    created_at_ns=clock.now_ns(),
                                    target_time_ms=target_t_ms,
                                    priority=1,
                                    cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
                                    metadata={
                                        "progress": round(p, 4),
                                        "repeat_index": k,
                                        "span_u": round(u, 4),
                                        "is_slider": True,
                                    },
                                )
                            )

                    # Tail Dwell Waypoint: Lock cursor to slider tail position until visual slider end
                    tail_x_osu, tail_y_osu = self._interpolate_slider_path(all_pts, 1.0, repeat_count, cum_lens=cum_lens)
                    tail_sx, tail_sy = self.preprocessor.osu_to_screen(tail_x_osu, tail_y_osu)
                    dwell_ns = timing_calibrator.calculate_dispatch_time_ns(end_t_ms, timeline_anchor_ns)
                    intents.append(
                        ActionIntent(
                            action_id=f"act_{w_obj.id}_slide_dwell",
                            action_type=ActionType.MOUSE_MOVE,
                            execute_at_ns=dwell_ns,
                            deadline_ns=end_dispatch_ns + tolerance_ns,
                            mode=GameMode.STANDARD,
                            target_id=w_obj.id,
                            x=float(tail_sx),
                            y=float(tail_sy),
                            created_at_ns=clock.now_ns(),
                            target_time_ms=target_t_ms,
                            priority=1,
                            cursor_owner=CursorOwnerState.SLIDER_FOLLOW,
                            metadata={"is_slider": True, "is_dwell": True},
                        )
                    )

                # Slider End: Cleanly release Key at slider tail
                intents.append(
                    ActionIntent(
                        action_id=f"act_{w_obj.id}_up",
                        action_type=ActionType.KEY_UP,
                        execute_at_ns=end_dispatch_ns,
                        deadline_ns=end_dispatch_ns + tolerance_ns,
                        mode=GameMode.STANDARD,
                        target_id=w_obj.id,
                        key=key,
                        button=None,
                        created_at_ns=clock.now_ns(),
                        target_time_ms=target_t_ms,
                        priority=3,
                        metadata={"is_slider": True},
                    )
                )

            elif isinstance(hit, Spinner):
                end_t_ms = target_t_ms + duration_ms
                end_dispatch_ns = timing_calibrator.calculate_dispatch_time_ns(end_t_ms, timeline_anchor_ns)

                # Spinner Start: Key Down
                intents.append(
                    ActionIntent(
                        action_id=f"act_{w_obj.id}_spin_down",
                        action_type=ActionType.KEY_DOWN,
                        execute_at_ns=dispatch_ns,
                        deadline_ns=deadline_ns,
                        mode=GameMode.STANDARD,
                        target_id=w_obj.id,
                        key=key,
                        button=None,
                        created_at_ns=clock.now_ns(),
                        target_time_ms=target_t_ms,
                        priority=3,
                    )
                )

                center_sx, center_sy = self.preprocessor.osu_to_screen(256.0, 192.0)
                if allow_mouse:
                    # 475 RPM near-maximum rotation speed (omega = 49.74 rad/s)
                    # Compact radius (32px) reduces physical travel distance and input flood
                    radius = 32.0
                    step_dt_ms = 16.0
                    spin_steps = max(6, int(duration_ms / step_dt_ms))
                    spin_step_ns = int((duration_ms / spin_steps) * 1_000_000)
                    step_dt_s = (duration_ms / spin_steps) / 1000.0
                    angular_vel = 2.0 * math.pi * (475.0 / 60.0)  # ~49.74 rad/s
                    d_theta = angular_vel * step_dt_s

                    # Early-exit buffer: if next note exists shortly after spinner, exit early to pre-aim
                    curr_idx = sorted_objects.index(w_obj) if w_obj in sorted_objects else -1
                    has_immediate_next = False
                    if 0 <= curr_idx < len(sorted_objects) - 1:
                        nxt_t = getattr(sorted_objects[curr_idx + 1].hit_object, "time_ms", 0.0)
                        if (nxt_t - end_t_ms) < 350.0:
                            has_immediate_next = True

                    early_exit_steps = max(1, int(min(35.0, duration_ms * 0.08) / max(0.001, step_dt_s * 1000.0))) if has_immediate_next else 0
                    effective_steps = max(4, spin_steps - early_exit_steps)

                    curr_angle = 0.0
                    for s in range(effective_steps):
                        curr_angle = (curr_angle + d_theta) % (2.0 * math.pi)
                        spin_x = center_sx + radius * math.cos(curr_angle)
                        spin_y = center_sy + radius * math.sin(curr_angle)
                        s_move_ns = dispatch_ns + (s * spin_step_ns)

                        intents.append(
                            ActionIntent(
                                action_id=f"act_{w_obj.id}_spin_move_{s}",
                                action_type=ActionType.MOUSE_MOVE,
                                execute_at_ns=s_move_ns,
                                deadline_ns=s_move_ns + 50_000_000,
                                mode=GameMode.STANDARD,
                                target_id=w_obj.id,
                                x=float(spin_x),
                                y=float(spin_y),
                                created_at_ns=clock.now_ns(),
                                target_time_ms=target_t_ms,
                                priority=2,
                                cursor_owner=CursorOwnerState.SPINNER_CONTROL,
                            )
                        )

                # Spinner End: Key Up
                intents.append(
                    ActionIntent(
                        action_id=f"act_{w_obj.id}_spin_up",
                        action_type=ActionType.KEY_UP,
                        execute_at_ns=end_dispatch_ns,
                        deadline_ns=end_dispatch_ns + tolerance_ns,
                        mode=GameMode.STANDARD,
                        target_id=w_obj.id,
                        key=key,
                        button=None,
                        created_at_ns=clock.now_ns(),
                        target_time_ms=target_t_ms,
                        priority=3,
                    )
                )

            else:
                # Regular Hit Circle: Key Tap (z/x) - Keyboard only
                intents.append(
                    ActionIntent(
                        action_id=f"act_{w_obj.id}",
                        action_type=ActionType.KEY_TAP,
                        execute_at_ns=effective_dispatch_ns,
                        deadline_ns=deadline_ns,
                        mode=GameMode.STANDARD,
                        target_id=w_obj.id,
                        x=float(target_sx),
                        y=float(target_sy),
                        key=key,
                        button=None,
                        created_at_ns=clock.now_ns(),
                        target_time_ms=target_t_ms,
                        priority=3,
                        metadata={"tap_hold_ms": tap_hold},
                    )
                )

            if not allow_mouse or target_t_ms >= earliest_move_ms or post_slider_rescued:
                sim_cursor_x = obj_end_x
                sim_cursor_y = obj_end_y
            if isinstance(hit, Slider):
                sim_last_is_slider = True
                sim_last_end_time_ms = max(sim_last_end_time_ms, safe_tail_ms)
            elif isinstance(hit, Spinner):
                sim_last_is_slider = False
                has_next = (w_obj in sorted_objects and sorted_objects.index(w_obj) < len(sorted_objects) - 1 and (getattr(sorted_objects[sorted_objects.index(w_obj) + 1].hit_object, "time_ms", 0.0) - end_t_ms) < 350.0)
                spinner_free_t = (end_t_ms - min(35.0, duration_ms * 0.08)) if (allow_mouse and has_next) else end_t_ms
                sim_last_end_time_ms = max(sim_last_end_time_ms, spinner_free_t)
            else:
                sim_last_is_slider = False
                dwell_ms = min(3.0, max(0.5, delta_t * 0.04)) if delta_t > 0 else 2.0
                sim_last_end_time_ms = max(sim_last_end_time_ms, target_t_ms + dwell_ms)

        # Update baseline key state from sorted objects processed in this horizon
        if sorted_objects:
            self._last_assigned_key = sim_last_key
            self._last_assigned_time_ms = sim_last_time_ms

        for s_obj in sorted_objects:
            hit_t = getattr(s_obj.hit_object, "time_ms", 0.0)
            dur = getattr(s_obj.hit_object, "duration_ms", 0.0) or 0.0
            if (hit_t + dur) <= current_time_ms:
                self._last_completed_time_ms = max(self._last_completed_time_ms, hit_t + dur)
                if isinstance(s_obj.hit_object, Slider):
                    p_end_x, p_end_y = self._interpolate_slider_path(
                        getattr(s_obj.hit_object, "_cached_path", [(s_obj.hit_object.x, s_obj.hit_object.y)]),
                        1.0,
                        getattr(s_obj.hit_object, "repeat_count", 1),
                        cum_lens=getattr(s_obj.hit_object, "_cached_cum_lens", None),
                    )
                    self._cursor_x = p_end_x
                    self._cursor_y = p_end_y
                else:
                    self._cursor_x = getattr(s_obj.hit_object, "x", self._cursor_x)
                    self._cursor_y = getattr(s_obj.hit_object, "y", self._cursor_y)

        return intents
