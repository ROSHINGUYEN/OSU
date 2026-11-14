"""
Central Input Queue with Deadline-Aware Priority Heap.
Maintains order of execution, rejects stale actions, and handles backpressure.
"""

import heapq
import threading
from typing import List, Optional, Tuple

from src.gameplay.osu.input.action_model import ActionIntent, ActionType, CursorOwnerState


class CentralInputQueue:
    """Thread-safe priority queue ordered by execution deadline."""

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._heap: List[Tuple[int, int, int, ActionIntent]] = []  # (execute_at_ns, priority, seq, ActionIntent)
        self._enqueued_ids = set()
        self._lock = threading.Lock()
        self._counter = 0
        self._stale_rejected_count = 0

    @property
    def stale_rejected_count(self) -> int:
        with self._lock:
            return self._stale_rejected_count

    def push(self, action: ActionIntent) -> bool:
        """Add an action intent to the priority queue with deduplication."""
        with self._lock:
            if action.action_id in self._enqueued_ids:
                return False
            if len(self._heap) >= self._max_size:
                return False
            self._enqueued_ids.add(action.action_id)
            self._counter += 1
            heapq.heappush(self._heap, (action.execute_at_ns, action.priority, self._counter, action))
            return True

    def push_group(self, actions: List[ActionIntent]) -> bool:
        """Atomically push a list of action intents. If full or duplicate detected, reject entire group."""
        with self._lock:
            if not actions:
                return True
            for action in actions:
                if action.action_id in self._enqueued_ids:
                    return False
            if len(self._heap) + len(actions) > self._max_size:
                return False
            for action in actions:
                self._enqueued_ids.add(action.action_id)
                self._counter += 1
                heapq.heappush(self._heap, (action.execute_at_ns, action.priority, self._counter, action))
            return True

    def peek(self) -> Optional[ActionIntent]:
        """View earliest action without removing it."""
        with self._lock:
            if not self._heap:
                return None
            return self._heap[0][3]

    def pop_due(self, current_ns: int) -> Optional[ActionIntent]:
        """Pop the earliest action if current_ns >= execute_at_ns, discarding any stale actions."""
        with self._lock:
            while self._heap:
                execute_at, priority, seq, action = self._heap[0]
                # Check if stale
                if action.is_stale(current_ns):
                    heapq.heappop(self._heap)
                    self._enqueued_ids.discard(action.action_id)
                    if action.action_type in (ActionType.KEY_UP, ActionType.MOUSE_UP):
                        # SAFETY: Never discard release actions; execute them immediately to prevent stuck keys!
                        return action
                    self._stale_rejected_count += 1
                    continue

                if current_ns >= execute_at:
                    heapq.heappop(self._heap)
                    self._enqueued_ids.discard(action.action_id)

                    # Mouse Move Coalescing:
                    # If this is a mouse move and multiple intermediate waypoints are already overdue,
                    # skip obsolete intermediate waypoints and jump directly to the freshest overdue mouse move.
                    # Coalescing is only valid within a single movement sequence: the same target_id AND the
                    # same cursor owner. A waypoint from Slider B must never replace a necessary waypoint from
                    # Slider A, and an unrelated circle's move must not be swallowed by another target's waypoint.
                    if action.action_type == ActionType.MOUSE_MOVE:
                        while self._heap:
                            next_exec, next_prio, next_seq, next_act = self._heap[0]
                            is_same_sequence = (
                                next_act.action_type == ActionType.MOUSE_MOVE
                                and current_ns >= next_exec
                                and bool(action.target_id)
                                and action.target_id == next_act.target_id
                                and action.cursor_owner == next_act.cursor_owner
                                and action.cursor_owner not in (None, CursorOwnerState.IDLE)
                            )
                            if is_same_sequence:
                                heapq.heappop(self._heap)
                                self._enqueued_ids.discard(next_act.action_id)
                                action = next_act
                            else:
                                break

                    return action
                break
            return None

    def prune_stale(self, current_ns: int) -> int:
        """Remove all stale actions where deadline_ns < current_ns, preserving release actions."""
        with self._lock:
            new_heap = []
            dropped = 0
            for item in self._heap:
                execute_at, priority, seq, action = item
                if action.is_stale(current_ns):
                    if action.action_type in (ActionType.KEY_UP, ActionType.MOUSE_UP):
                        # SAFETY: Never discard release actions; clamp execute_at_ns to current_ns so pop_due executes them immediately
                        action.execute_at_ns = min(action.execute_at_ns, current_ns)
                        new_heap.append((action.execute_at_ns, priority, seq, action))
                    else:
                        dropped += 1
                else:
                    new_heap.append(item)
            heapq.heapify(new_heap)
            self._heap = new_heap
            self._enqueued_ids = {item[3].action_id for item in self._heap}
            self._stale_rejected_count += dropped
            return dropped

    def clear(self) -> None:
        """Instantly wipe queue (used by Emergency Stop and Active Revocation)."""
        with self._lock:
            self._heap.clear()
            self._enqueued_ids.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)


input_queue = CentralInputQueue()
