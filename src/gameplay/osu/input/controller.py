"""
Central Input Controller for Shiori AI osu! Gameplay.
Single point of contact for OS input execution, state tracking, and stuck-key prevention.
"""

import heapq
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode, InputState, state_store
from src.gameplay.osu.input.action_model import (
    ActionIntent,
    ActionType,
    CursorOwnerState,
    InputStateStatus,
)
from src.gameplay.osu.input.backends.base import BaseInputBackend
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.input.backends.win32_sendinput import Win32SendInputBackend

logger = logging.getLogger("Shiori.Osu.InputController")

# Standard osu! Key Controller


class CentralInputController:
    """Controls OS-level mouse and keyboard inputs with four-tier verification and stuck-key watchdog."""

    def __init__(
        self,
        backend: Optional[BaseInputBackend] = None,
        max_stuck_duration_ms: float = 3000.0,
    ):
        if backend is not None:
            self._backend = backend
        else:
            try:
                import os
                if os.name == "nt":
                    self._backend = Win32SendInputBackend()
                else:
                    self._backend = MockInputBackend()
            except Exception as e:
                logger.warning(f"Failed to initialize Win32SendInputBackend: {e}, falling back to Mock.")
                self._backend = MockInputBackend()
        self._max_stuck_duration_ns: int = int(max_stuck_duration_ms * 1_000_000)
        self._tap_hold_ns: int = int(0.012 * 1_000_000_000)  # 12ms ultra-fast tap hold for high-BPM 260+ stream speed
        self._key_down_times: Dict[str, int] = {}
        self._last_key_up_times: Dict[str, int] = {}
        self._mouse_down_times: Dict[str, int] = {}
        self._cursor_pos: tuple = (0.0, 0.0)
        self.auto_aim_enabled: bool = True
        self.enable_mouse_clicks: bool = not isinstance(self._backend, MockInputBackend)
        self.enable_companion_keys: bool = not isinstance(self._backend, MockInputBackend)
        self._lock = threading.Lock()
        self._stuck_key_warnings: int = 0
        self._locked_slider_keys: set = set()
        self._locked_slider_buttons: set = set()
        self._active_slider_owner: Optional[str] = None
        self._active_slider_key: Optional[str] = None
        self._remapped_action_keys: Dict[str, str] = {}

        # Cursor ownership state machine (P0 §7.3)
        # Explicit ownership with acquisition, validation, release, cancellation rules.
        self._cursor_owner_state: CursorOwnerState = CursorOwnerState.IDLE
        self._cursor_owner_target: Optional[str] = None       # target_id that currently owns the cursor
        self._cursor_owner_sequence: Optional[str] = None     # action_id sequence that owns the cursor
        self._cursor_owner_acquired_at_ns: Optional[int] = None
        # Sequence IDs of cancelled/completed targets whose queued waypoints must be rejected
        self._invalidated_sequences: set = set()

        self._release_heap: List[Tuple[Any, ...]] = []
        self._wake_event = threading.Event()
        self._worker_running = True
        self._worker_thread = threading.Thread(
            target=self._release_worker_loop,
            daemon=True,
            name="ShioriKeyReleaseWorker",
        )
        self._worker_thread.start()

    def _release_worker_loop(self) -> None:
        """Background worker ensuring keys and mouse buttons tapped are held for >= 12ms and cleanly released."""
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
        except Exception:
            pass
        while self._worker_running:
            keys_to_up: List[str] = []
            buttons_to_up: List[str] = []
            with self._lock:
                now_ns = clock.now_ns()
                while self._release_heap and self._release_heap[0][0] <= now_ns:
                    item = heapq.heappop(self._release_heap)
                    if len(item) == 2:
                        _, key = item
                        keys_to_up.append(key)
                    elif len(item) == 3:
                        _, itype, name = item
                        if itype == "mouse":
                            buttons_to_up.append(name)
                        else:
                            keys_to_up.append(name)

                for key in keys_to_up:
                    # SAFETY: Never release a key that is currently locked by an active slider!
                    if key in self._locked_slider_keys:
                        continue
                    self._backend.key_up(key)
                    self._last_key_up_times[key] = now_ns
                    if key in self._key_down_times:
                        del self._key_down_times[key]

                for btn in buttons_to_up:
                    if btn in self._locked_slider_buttons:
                        continue
                    self._backend.mouse_up(btn)
                    if btn in self._mouse_down_times:
                        del self._mouse_down_times[btn]

                if keys_to_up or buttons_to_up:
                    self._sync_to_store()

                if not self._release_heap:
                    next_timeout = 0.05
                else:
                    diff_s = (self._release_heap[0][0] - now_ns) / 1_000_000_000.0
                    next_timeout = max(0.001, min(0.02, diff_s))

            self._wake_event.wait(timeout=next_timeout)
            self._wake_event.clear()

    @property
    def backend(self) -> BaseInputBackend:
        return self._backend

    @property
    def cursor_pos(self) -> tuple:
        return self._cursor_pos

    def set_backend(self, backend: BaseInputBackend) -> None:
        with self._lock:
            self._backend = backend
            self.enable_mouse_clicks = not isinstance(backend, MockInputBackend)
            self.enable_companion_keys = not isinstance(backend, MockInputBackend)

    def execute_action(self, action: ActionIntent) -> bool:
        """Execute an action intent through the underlying OS backend."""
        with self._lock:
            self.check_stuck_keys()

            if not getattr(self, "auto_aim_enabled", True) and action.action_type in (
                ActionType.MOUSE_MOVE,
                ActionType.MOUSE_DOWN,
                ActionType.MOUSE_UP,
                ActionType.MOUSE_CLICK,
            ):
                action.status = InputStateStatus.CANCELLED
                return True

            action.status = InputStateStatus.SUBMISSION
            success = False

            if action.action_type == ActionType.MOUSE_MOVE:
                # 1. Stale / invalidated target suppression:
                # If a target was cancelled or ended, drop subsequent queued waypoints.
                if action.target_id and action.target_id in self._invalidated_sequences:
                    logger.debug(
                        f"Suppressed waypoint {action.action_id} for invalidated target {action.target_id}"
                    )
                    action.status = InputStateStatus.CANCELLED
                    return True

                # STRICT SLIDER CURSOR LOCK (P0 §7.3):
                # If a slider currently owns the cursor, reject/suppress any foreign MOUSE_MOVE
                # (e.g. from an upcoming hit circle or pre-aim trajectory) that attempts to displace the mouse!
                active_owner = self._active_slider_owner or (
                    self._cursor_owner_target if self._cursor_owner_state == CursorOwnerState.SLIDER_FOLLOW else None
                )
                if (self._locked_slider_keys and active_owner is not None) or (
                    self._cursor_owner_state == CursorOwnerState.SLIDER_FOLLOW and active_owner is not None
                ):
                    is_slider_movement = (
                        action.cursor_owner == CursorOwnerState.SLIDER_FOLLOW
                        or action.target_id == active_owner
                        or action.cursor_owner == CursorOwnerState.RECOVERY
                    )
                    if not is_slider_movement:
                        logger.debug(
                            f"Suppressed foreign cursor move {action.action_id} (owner={action.cursor_owner}, "
                            f"target={action.target_id}) while slider {active_owner} is active"
                        )
                        action.status = InputStateStatus.CANCELLED
                        return True

                if action.x is not None and action.y is not None:
                    success = self._backend.move_cursor(int(action.x), int(action.y))
                    if success:
                        self._cursor_pos = (action.x, action.y)
                        if "slide_dwell" in action.action_id or action.metadata.get("is_slider_dwell"):
                            if self._active_slider_owner:
                                self._invalidated_sequences.add(self._active_slider_owner)
                            self._active_slider_owner = None
                            if self._cursor_owner_target == action.target_id:
                                self._cursor_owner_state = CursorOwnerState.IDLE
                                self._cursor_owner_target = None
                                self._cursor_owner_acquired_at_ns = None

            elif action.action_type == ActionType.MOUSE_DOWN:
                btn = getattr(action, "button", "left") or "left"
                success = self._backend.mouse_down(btn)
                if success:
                    self._mouse_down_times[btn] = clock.now_ns()

            elif action.action_type == ActionType.MOUSE_UP:
                btn = getattr(action, "button", "left") or "left"
                success = self._backend.mouse_up(btn)
                if btn in self._mouse_down_times:
                    del self._mouse_down_times[btn]

            elif action.action_type == ActionType.MOUSE_CLICK:
                btn = getattr(action, "button", "left") or "left"
                success = self._backend.mouse_down(btn)
                if success:
                    now = clock.now_ns()
                    self._mouse_down_times[btn] = now
                    heapq.heappush(self._release_heap, (now + self._tap_hold_ns, "mouse", btn))
                    self._wake_event.set()

            elif action.action_type == ActionType.KEY_DOWN:
                if action.key:
                    now_ns = clock.now_ns()
                    # Hardware Debounce & Slider Protection:
                    # If key is locked by active slider or released < 20ms ago, prefer alternate key if free
                    alt_key = "x" if action.key == "z" else ("z" if action.key == "x" else None)
                    if alt_key and alt_key not in self._locked_slider_keys and alt_key not in self._key_down_times:
                        is_locked = action.key in self._locked_slider_keys
                        is_debounce_risk = (now_ns - self._last_key_up_times.get(action.key, 0)) < 20_000_000
                        if is_locked or is_debounce_risk:
                            action.key = alt_key
                            if action.target_id:
                                self._remapped_action_keys[action.target_id] = alt_key

                    self._locked_slider_keys.add(action.key)
                    if action.metadata.get("is_slider") or (action.target_id and "slide" in action.action_id):
                        self._active_slider_owner = action.target_id
                        self._active_slider_key = action.key
                        self._cursor_owner_state = CursorOwnerState.SLIDER_FOLLOW
                        self._cursor_owner_target = action.target_id
                        self._cursor_owner_acquired_at_ns = now_ns

                    # Safety: Purge any pending release timers in _release_heap for this key
                    if self._release_heap:
                        self._release_heap = [
                            item for item in self._release_heap
                            if not (
                                (len(item) == 2 and item[1] == action.key)
                                or (len(item) == 3 and item[1] == "key" and item[2] == action.key)
                            )
                        ]
                        heapq.heapify(self._release_heap)

                    if action.key in self._key_down_times:
                        # Key was already down: immediately release it first to ensure a crisp down edge transition
                        self._backend.key_up(action.key)
                        del self._key_down_times[action.key]
                    success = self._backend.key_down(action.key)
                    if success:
                        self._key_down_times[action.key] = clock.now_ns()

                btn = getattr(action, "button", None)
                if btn and getattr(self, "enable_mouse_clicks", False):
                    self._locked_slider_buttons.add(btn)
                    if btn in self._mouse_down_times:
                        self._backend.mouse_up(btn)
                        del self._mouse_down_times[btn]
                    m_ok = self._backend.mouse_down(btn)
                    if m_ok:
                        self._mouse_down_times[btn] = clock.now_ns()
                    success = success or m_ok

            elif action.action_type == ActionType.KEY_UP:
                # Resolve exact pressed key if remapped during KEY_DOWN or owned by active slider
                if action.target_id and action.target_id in self._remapped_action_keys:
                    action.key = self._remapped_action_keys.pop(action.target_id)
                elif action.target_id == self._active_slider_owner and self._active_slider_key:
                    action.key = self._active_slider_key

                if action.key:
                    self._locked_slider_keys.discard(action.key)
                    if action.target_id == self._active_slider_owner or not self._locked_slider_keys:
                        if self._active_slider_owner:
                            self._invalidated_sequences.add(self._active_slider_owner)
                        self._active_slider_owner = None
                        self._active_slider_key = None
                        if self._cursor_owner_target == action.target_id:
                            self._cursor_owner_state = CursorOwnerState.IDLE
                            self._cursor_owner_target = None
                            self._cursor_owner_acquired_at_ns = None

                    # Safety: Purge any release heap timer for this key
                    if self._release_heap:
                        self._release_heap = [
                            item for item in self._release_heap
                            if not (
                                (len(item) == 2 and item[1] == action.key)
                                or (len(item) == 3 and item[1] == "key" and item[2] == action.key)
                            )
                        ]
                        heapq.heapify(self._release_heap)
                    success = self._backend.key_up(action.key)
                    self._last_key_up_times[action.key] = clock.now_ns()
                    if action.key in self._key_down_times:
                        del self._key_down_times[action.key]

                btn = getattr(action, "button", None)
                if btn and getattr(self, "enable_mouse_clicks", False):
                    self._locked_slider_buttons.discard(btn)
                    m_ok = self._backend.mouse_up(btn)
                    if btn in self._mouse_down_times:
                        del self._mouse_down_times[btn]
                    success = success or m_ok

            elif action.action_type == ActionType.KEY_TAP:
                hold_ns = int(action.metadata.get("tap_hold_ms", 0) * 1_000_000) or self._tap_hold_ns
                if action.key:
                    now_ns = clock.now_ns()
                    # SLIDER KEY PROTECTION & HARDWARE DEBOUNCE:
                    # If target key is locked by active slider or released < 20ms ago, remap to alternate key
                    alt_key = "x" if action.key == "z" else ("z" if action.key == "x" else None)
                    if alt_key and alt_key not in self._locked_slider_keys and alt_key not in self._key_down_times:
                        is_locked = action.key in self._locked_slider_keys
                        is_debounce_risk = (now_ns - self._last_key_up_times.get(action.key, 0)) < 20_000_000
                        if is_locked or is_debounce_risk:
                            action.key = alt_key
                    elif action.key in self._locked_slider_keys:
                        # If target key is locked by active slider, prefer to tap with alternate key rather than dropping the note!
                        if alt_key:
                            action.key = alt_key

                    # Purge any old pending release timers for this key
                    if self._release_heap:
                        keys_to_purge = {action.key}
                        self._release_heap = [
                            item for item in self._release_heap
                            if not (
                                (len(item) == 2 and item[1] in keys_to_purge)
                                or (len(item) == 3 and item[1] == "key" and item[2] in keys_to_purge)
                            )
                        ]
                        heapq.heapify(self._release_heap)

                    if action.key in self._key_down_times:
                        # Key was already down: if not a locked slider, release it first so this tap registers as a fresh stroke
                        if action.key not in self._locked_slider_keys:
                            self._backend.key_up(action.key)
                            del self._key_down_times[action.key]
                    success = self._backend.key_down(action.key)
                    if success:
                        now = clock.now_ns()
                        self._key_down_times[action.key] = now
                        heapq.heappush(self._release_heap, (now + hold_ns, action.key))
                        self._wake_event.set()

                btn = getattr(action, "button", None)
                if btn and getattr(self, "enable_mouse_clicks", False):
                    # SLIDER BUTTON PROTECTION: If target button is locked by active slider, remap to alternate button
                    if btn in self._locked_slider_buttons:
                        alt_btn = "right" if btn == "left" else "left"
                        if alt_btn not in self._locked_slider_buttons:
                            btn = alt_btn

                    # Purge any old pending release timers for this mouse button
                    if self._release_heap:
                        self._release_heap = [item for item in self._release_heap if not (len(item) == 3 and item[1] == "mouse" and item[2] == btn)]
                        heapq.heapify(self._release_heap)

                    if btn in self._mouse_down_times:
                        self._backend.mouse_up(btn)
                        del self._mouse_down_times[btn]
                    m_ok = self._backend.mouse_down(btn)
                    if m_ok:
                        now = clock.now_ns()
                        self._mouse_down_times[btn] = now
                        heapq.heappush(self._release_heap, (now + hold_ns, "mouse", btn))
                        self._wake_event.set()
                    success = success or m_ok

            self._sync_to_store()
            return success

    def tap_key(self, key: str) -> bool:
        """Convenience method to execute a clean KEY_TAP with automatic timed release."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"tap_{key}_{now}",
            action_type=ActionType.KEY_TAP,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            key=key,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def key_down(self, key: str) -> bool:
        """Convenience method to press and hold down a keyboard key."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"kdown_{key}_{now}",
            action_type=ActionType.KEY_DOWN,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            key=key,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def key_up(self, key: str) -> bool:
        """Convenience method to release a held keyboard key."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"kup_{key}_{now}",
            action_type=ActionType.KEY_UP,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            key=key,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def mouse_down(self, button: str = "left") -> bool:
        """Convenience method to press down a mouse button."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"mdown_{button}_{now}",
            action_type=ActionType.MOUSE_DOWN,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            button=button,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def mouse_up(self, button: str = "left") -> bool:
        """Convenience method to release a mouse button."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"mup_{button}_{now}",
            action_type=ActionType.MOUSE_UP,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            button=button,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def mouse_click(self, button: str = "left") -> bool:
        """Convenience method to execute a clean MOUSE_CLICK with automatic timed release."""
        now = clock.now_ns()
        intent = ActionIntent(
            action_id=f"mclick_{button}_{now}",
            action_type=ActionType.MOUSE_CLICK,
            execute_at_ns=now,
            deadline_ns=now + 50_000_000,
            button=button,
            created_at_ns=now,
        )
        return self.execute_action(intent)

    def move_cursor(self, screen_x: int, screen_y: int) -> bool:
        """Convenience method to move OS cursor."""
        if not getattr(self, "auto_aim_enabled", True):
            return False
        with self._lock:
            success = self._backend.move_cursor(int(screen_x), int(screen_y))
            if success:
                self._cursor_pos = (float(screen_x), float(screen_y))
                self._sync_to_store()
            return success

    def check_stuck_keys(self) -> int:
        """Watchdog to auto-release any key or mouse button held longer than max_stuck_duration_ms."""
        now_ns = clock.now_ns()
        stuck_keys = []
        for key, down_time in self._key_down_times.items():
            threshold_ns = (self._max_stuck_duration_ns * 5) if (key in self._locked_slider_keys and self._max_stuck_duration_ns >= 1_000_000_000) else self._max_stuck_duration_ns
            if now_ns - down_time > threshold_ns:
                stuck_keys.append(key)

        for key in stuck_keys:
            logger.warning(f"Stuck key watchdog triggered: releasing '{key}' (held > max duration).")
            self._backend.key_up(key)
            self._locked_slider_keys.discard(key)
            del self._key_down_times[key]
            self._stuck_key_warnings += 1

        stuck_buttons = []
        for btn, down_time in self._mouse_down_times.items():
            if now_ns - down_time > self._max_stuck_duration_ns:
                stuck_buttons.append(btn)

        for btn in stuck_buttons:
            logger.warning(f"Stuck mouse button watchdog triggered: releasing '{btn}'.")
            self._backend.mouse_up(btn)
            self._locked_slider_buttons.discard(btn)
            del self._mouse_down_times[btn]
            self._stuck_key_warnings += 1

        return len(stuck_keys) + len(stuck_buttons)

    def cancel_active_slider(self, target_id: Optional[str] = None) -> None:
        """Explicitly cancels active slider lock and safely releases associated keys and ownership."""
        with self._lock:
            cancelled_target = target_id or self._active_slider_owner or self._cursor_owner_target
            if cancelled_target:
                self._invalidated_sequences.add(cancelled_target)
            if target_id is None or target_id == self._active_slider_owner or target_id == self._cursor_owner_target:
                if target_id:
                    self._remapped_action_keys.pop(target_id, None)
                if self._active_slider_key:
                    self._locked_slider_keys.discard(self._active_slider_key)
                    self._backend.key_up(self._active_slider_key)
                    if self._active_slider_key in self._key_down_times:
                        del self._key_down_times[self._active_slider_key]
                self._active_slider_owner = None
                self._active_slider_key = None
                self._cursor_owner_state = CursorOwnerState.IDLE
                self._cursor_owner_target = None
                self._cursor_owner_acquired_at_ns = None
                self._sync_to_store()

    def emergency_release_all(self) -> None:
        """Instantly release all keys and mouse buttons."""
        with self._lock:
            had_inputs = bool(self._key_down_times or self._mouse_down_times or self._release_heap)
            self._release_heap.clear()
            self._backend.release_all()
            self._key_down_times.clear()
            self._mouse_down_times.clear()
            self._locked_slider_keys.clear()
            self._locked_slider_buttons.clear()
            self._active_slider_owner = None
            self._active_slider_key = None
            self._cursor_owner_state = CursorOwnerState.IDLE
            self._cursor_owner_target = None
            self._cursor_owner_acquired_at_ns = None
            self._invalidated_sequences.clear()
            self._remapped_action_keys.clear()
            self._sync_to_store()
            if had_inputs:
                logger.info("Emergency release completed: all inputs neutralized.")

    @property
    def cursor_owner_state(self) -> CursorOwnerState:
        with self._lock:
            return self._cursor_owner_state

    @property
    def cursor_owner_target(self) -> Optional[str]:
        with self._lock:
            return self._cursor_owner_target

    def invalidate_target(self, target_id: str) -> None:
        """Mark a target_id as invalidated/completed so that subsequent queued waypoints are dropped."""
        with self._lock:
            if not target_id:
                return
            self._invalidated_sequences.add(target_id)
            if self._cursor_owner_target == target_id:
                self._cursor_owner_state = CursorOwnerState.IDLE
                self._cursor_owner_target = None
                self._cursor_owner_acquired_at_ns = None
            if len(self._invalidated_sequences) > 500:
                self._invalidated_sequences = set(list(self._invalidated_sequences)[-250:])

    def _sync_to_store(self) -> None:
        key_states = {k: "DOWN" for k in self._key_down_times}
        for btn in self._mouse_down_times:
            key_states[f"mouse_{btn}"] = "DOWN"
        state_store.mutate_domain(
            "input",
            lambda curr: InputState(
                key_states=key_states,
                cursor_pos=self._cursor_pos,
                stuck_key_warnings=self._stuck_key_warnings,
                updated_at_ns=clock.now_ns(),
            ),
            caller="input_controller",
        )


input_controller = CentralInputController()
