"""
High-Frequency Data Channels with Circular Ring Buffers.
Avoids passing heavy raw frames and high-rate telemetry through the Event Bus.
"""

import threading
from collections import deque
from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


class DataChannel(Generic[T]):
    """
    Thread-safe circular ring buffer data channel for high-frequency streams.
    Oldest data is automatically overwritten when capacity is reached.
    """

    def __init__(self, name: str, capacity: int = 120):
        self._name: str = name
        self._capacity: int = capacity
        self._buffer: deque[T] = deque(maxlen=capacity)
        self._lock: threading.Lock = threading.Lock()
        self._total_pushed: int = 0
        self._dropped_count: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_pushed(self) -> int:
        return self._total_pushed

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    def push(self, item: T) -> None:
        """Push an item to the channel. Drops oldest item if full."""
        with self._lock:
            if len(self._buffer) == self._capacity:
                self._dropped_count += 1
            self._buffer.append(item)
            self._total_pushed += 1

    def get_latest(self) -> Optional[T]:
        """Get the most recent item without removing it."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]

    def get_recent(self, n: int = 1) -> List[T]:
        """Get up to n most recent items in chronological order."""
        with self._lock:
            if not self._buffer:
                return []
            n_items = min(n, len(self._buffer))
            return list(self._buffer)[-n_items:]

    def clear(self) -> None:
        """Clear the channel buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# Global standard channels
frame_channel: DataChannel[Any] = DataChannel("frame_channel", capacity=60)
telemetry_channel: DataChannel[Any] = DataChannel("telemetry_channel", capacity=500)
