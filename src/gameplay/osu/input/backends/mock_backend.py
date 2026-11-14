"""
Mock Input Backend for Unit Testing and Headless Simulation.
"""

from typing import Any, List, Set, Tuple
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.input.backends.base import BaseInputBackend


class MockInputBackend(BaseInputBackend):
    """Safe in-memory input recorder without OS interactions."""

    def __init__(self):
        self.cursor_pos: Tuple[int, int] = (0, 0)
        self.active_keys: Set[str] = set()
        self.active_mouse_buttons: Set[str] = set()
        self.history: List[Tuple[str, Any, int]] = []

    def move_cursor(self, screen_x: int, screen_y: int) -> bool:
        self.cursor_pos = (screen_x, screen_y)
        self.history.append(("move", (screen_x, screen_y), clock.now_ns()))
        return True

    def mouse_down(self, button: str = "left") -> bool:
        button = button.lower()
        self.active_mouse_buttons.add(button)
        self.history.append(("mouse_down", button, clock.now_ns()))
        return True

    def mouse_up(self, button: str = "left") -> bool:
        button = button.lower()
        if button in self.active_mouse_buttons:
            self.active_mouse_buttons.remove(button)
        self.history.append(("mouse_up", button, clock.now_ns()))
        return True

    def key_down(self, key: str) -> bool:
        key = key.lower()
        self.active_keys.add(key)
        self.history.append(("key_down", key, clock.now_ns()))
        return True

    def key_up(self, key: str) -> bool:
        key = key.lower()
        if key in self.active_keys:
            self.active_keys.remove(key)
        self.history.append(("key_up", key, clock.now_ns()))
        return True

    def release_all(self) -> None:
        for key in list(self.active_keys):
            self.key_up(key)
        self.active_keys.clear()
        for btn in list(self.active_mouse_buttons):
            self.mouse_up(btn)
        self.active_mouse_buttons.clear()

    def clear(self) -> None:
        self.cursor_pos = (0, 0)
        self.active_keys.clear()
        self.active_mouse_buttons.clear()
        self.history.clear()
