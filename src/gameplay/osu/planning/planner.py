"""
Rolling-Horizon Planner for Shiori AI osu! Gameplay.
Plans actions across a 500-1500ms lookahead horizon and commits near-term actions to the Input Queue.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src.gameplay.osu.brains.base import BaseGameplayBrain
from src.gameplay.osu.brains.standard_brain import StandardBrain
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, PlannerState, state_store
from src.gameplay.osu.input.action_model import ActionIntent
from src.gameplay.osu.input.queue import CentralInputQueue, input_queue
from src.gameplay.osu.perception.preprocessor import PlayfieldPreprocessor
from src.gameplay.osu.planning.utility import UtilityEvaluator
from src.gameplay.osu.planning.validator import ActionValidator, action_validator
from src.gameplay.osu.world.object_lifecycle import ObjectLifecycleState, WorldObject
from src.gameplay.osu.world.world_state import (
    WorldStateManager,
    world_state_manager,
)

logger = logging.getLogger("Shiori.Osu.Planner")


class RollingHorizonPlanner:
    """Orchestrates candidate generation, utility ranking, and rolling action commitment."""

    def __init__(
        self,
        horizon_ms: float = 1000.0,
        commit_window_ms: float = 150.0,
        queue: CentralInputQueue = input_queue,
        world_manager: WorldStateManager = world_state_manager,
        validator: ActionValidator = action_validator,
        preprocessor: Optional[PlayfieldPreprocessor] = None,
    ):
        self.horizon_ms = horizon_ms
        self.commit_window_ms = commit_window_ms
        self._queue = queue
        self._world_manager = world_manager
        self._validator = validator
        self._utility_eval = UtilityEvaluator()

        # Pure Standard gameplay brain
        self._brains: Dict[GameMode, BaseGameplayBrain] = {
            GameMode.STANDARD: StandardBrain(preprocessor=preprocessor),
        }

        self._replan_counter = 0
        self._committed_actions = set()
        if preprocessor is not None:
            self.set_preprocessor(preprocessor)

    def set_preprocessor(self, preprocessor: PlayfieldPreprocessor) -> None:
        """Propagate playfield coordinate preprocessor to standard brain."""
        for brain in self._brains.values():
            if hasattr(brain, "preprocessor"):
                brain.preprocessor = preprocessor

    def get_brain(self, mode: Optional[GameMode] = None) -> Optional[BaseGameplayBrain]:
        """Return the registered brain for the exact gameplay mode.

        Explicit mode-to-Brain mapping: an unsupported or unknown mode returns
        None instead of silently falling back to the Standard brain, so callers
        can distinguish "no brain" from "Standard brain".
        """
        if mode is None or mode == GameMode.UNKNOWN:
            return None
        brain = self._brains.get(mode, None)
        if brain is None:
            logger.warning("No gameplay brain registered for mode %s; this mode is not supported for planning.", mode)
        return brain

    def plan_cycle(
        self,
        current_time_ms: float,
        timeline_anchor_ns: int,
        mode: Optional[GameMode] = None,
        observed_cursor_pos: Optional[Tuple[float, float]] = None,
    ) -> List[ActionIntent]:
        """
        Executes a single rolling-horizon planning iteration:
        1. Reads active WorldObjects within [current_time, current_time + horizon_ms].
        2. Generates candidate actions via mode brain.
        3. Validates and commits near-term actions to the CentralInputQueue.
        """
        self._replan_counter += 1
        active_mode = mode or state_store.mode.mode
        brain = self.get_brain(active_mode)
        if brain is None:
            logger.debug("Cannot plan cycle: Brain not found or mode unknown (%s)", active_mode)
            return []

        # 1. Gather upcoming objects in planning horizon
        all_objects = list(self._world_manager.active_objects.values())
        horizon_end_ms = current_time_ms + self.horizon_ms

        horizon_objects = []
        for obj in all_objects:
            start_t = getattr(obj.hit_object, "time_ms", 0.0)
            dur = getattr(obj.hit_object, "duration_ms", 0.0) or 0.0
            raw_end = getattr(obj.hit_object, "end_time_ms", 0.0) or 0.0
            obj_end = raw_end if raw_end > start_t else (start_t + dur)
            if (current_time_ms - 50.0 <= obj_end) and (start_t <= horizon_end_ms):
                horizon_objects.append(obj)
        horizon_objects.sort(key=lambda o: o.hit_object.time_ms)

        if not horizon_objects:
            return []

        # 2. Generate candidates from brain (with real-time observed cursor feedback when available)
        if active_mode == GameMode.STANDARD and observed_cursor_pos is not None:
            raw_candidates = brain.plan_actions(
                current_time_ms,
                horizon_objects,
                timeline_anchor_ns,
                observed_cursor_pos=observed_cursor_pos,
            )
        else:
            raw_candidates = brain.plan_actions(current_time_ms, horizon_objects, timeline_anchor_ns)

        # 3. Validate & commit candidates (atomic per target sequence)
        now_ns = clock.now_ns()
        # Gameplay-clock "now" expressed in the same monotonic domain as the
        # anchor-relative execute_at/deadline times (never mix clock domains).
        gameplay_now_ns = timeline_anchor_ns + int(current_time_ms * 1_000_000)
        commit_threshold_ns = timeline_anchor_ns + int((current_time_ms + self.commit_window_ms) * 1_000_000)
        committed: List[ActionIntent] = []

        # Identify targets whose earliest action enters the commitment window
        targets_to_commit: set = set()
        for candidate in raw_candidates:
            if candidate.action_id in self._committed_actions:
                continue
            if candidate.execute_at_ns <= commit_threshold_ns and candidate.target_id:
                targets_to_commit.add(candidate.target_id)

        # Group not-yet-committed candidates by target for atomic (all-or-nothing) enqueue.
        # A gameplay target must never be marked ACTION_SCHEDULED unless its complete
        # action sequence (e.g. slider down + waypoints + up) is enqueued together.
        candidates_by_target: Dict[str, List[ActionIntent]] = {}
        loose_candidates: List[ActionIntent] = []
        for candidate in raw_candidates:
            if candidate.action_id in self._committed_actions:
                continue

            # Deadline-aware feasibility: never (re)commit an action whose deadline has
            # already passed. Scheduling an impossible/stale action as if it were valid
            # diverges internal state from actual input execution.
            if candidate.is_stale(gameplay_now_ns):
                logger.debug(
                    "Skipping stale/infeasible action %s for target %s (deadline %s already passed)",
                    candidate.action_id, candidate.target_id, candidate.deadline_ns,
                )
                continue

            # Commit if the action itself is due, or its parent target sequence is being committed
            should_commit = (
                (candidate.execute_at_ns <= commit_threshold_ns)
                or (candidate.target_id and candidate.target_id in targets_to_commit)
            )
            if not should_commit:
                continue

            if candidate.target_id:
                candidates_by_target.setdefault(candidate.target_id, []).append(candidate)
            else:
                loose_candidates.append(candidate)

        # Commit loose (non-target-bound) actions individually
        for candidate in loose_candidates:
            is_valid, reason = self._validator.validate(candidate)
            if not is_valid:
                logger.warning(f"Rejected invalid action {candidate.action_id}: {reason}")
                continue
            success = self._queue.push(candidate)
            if success:
                self._committed_actions.add(candidate.action_id)
                committed.append(candidate)

        # Commit each target's full action sequence atomically via push_group
        for target_id, candidates in candidates_by_target.items():
            # Deterministic order within a sequence: execution time, then priority
            candidates.sort(key=lambda c: (c.execute_at_ns, c.priority))

            # Validate every action before attempting any enqueue
            sequence_invalid = False
            for candidate in candidates:
                is_valid, reason = self._validator.validate(candidate)
                if not is_valid:
                    sequence_invalid = True
                    logger.warning(
                        "Rejected target %s sequence: action %s invalid (%s)",
                        target_id, candidate.action_id, reason,
                    )
                    break
            if sequence_invalid:
                continue

            # All-or-nothing enqueue: never mark a target scheduled unless its entire
            # sequence is queued. On failure the lifecycle is left untouched so the
            # planner can replan the target on the next cycle.
            if not self._queue.push_group(candidates):
                logger.warning(
                    "Target %s sequence not enqueued (queue full or duplicate action); "
                    "lifecycle left unscheduled for replanning.",
                    target_id,
                )
                continue

            for candidate in candidates:
                self._committed_actions.add(candidate.action_id)
                committed.append(candidate)
            # Transition object lifecycle to ACTION_SCHEDULED only once the complete
            # action sequence has been atomically enqueued.
            self._world_manager.transition_state(
                target_id, ObjectLifecycleState.ACTION_SCHEDULED
            )

        # 4. Sync planner state
        state_store.mutate_domain(
            "planner",
            lambda curr: PlannerState(
                current_plan=[c.action_id for c in committed],
                planning_horizon_ms=self.horizon_ms,
                replan_counter=self._replan_counter,
                updated_at_ns=now_ns,
            ),
            caller="planner",
        )

        return committed

    def reset(self) -> None:
        """Reset internal brains and counters."""
        self._replan_counter = 0
        self._committed_actions.clear()
        if hasattr(self._world_manager, "clear"):
            self._world_manager.clear()
        for brain in self._brains.values():
            brain.reset()


rolling_planner = RollingHorizonPlanner()
