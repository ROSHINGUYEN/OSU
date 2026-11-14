"""
Abstract OS Input Backend Interface.
"""

from abc import ABC, abstractmethod


class BaseInputBackend(ABC):
    """Abstract driver interface for sending mouse and keyboard inputs."""

    @abstractmethod
    def move_cursor(self, screen_x: int, screen_y: int) -> bool:
        """Move OS cursor to absolute screen coordinate."""
        pass

    @abstractmethod
    def mouse_down(self, button: str = "left") -> bool:
        """Press down a mouse button ('left' or 'right')."""
        pass

    @abstractmethod
    def mouse_up(self, button: str = "left") -> bool:
        """Release a mouse button ('left' or 'right')."""
        pass

    @abstractmethod
    def key_down(self, key: str) -> bool:
        """Press down a key."""
        pass

    @abstractmethod
    def key_up(self, key: str) -> bool:
        """Release a key."""
        pass

    @abstractmethod
    def release_all(self) -> None:
        """Emergency release of all tracked pressed keys."""
        pass
