"""
Asynchronous & Synchronous In-Memory Event Bus for Shiori AI osu! Gameplay System.
Decouples Perception, World State, Planning, Input, Safety, and Feedback domains.
"""

import asyncio
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Union

logger = logging.getLogger("Shiori.Osu.EventBus")


@dataclass(frozen=True)
class Event:
    """Standardized event envelope across the osu! gameplay pipeline."""
    event_type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp_ns: int = 0
    source: str = "system"
    session_id: str = "default_session"
    sequence_id: int = 0


HandlerType = Union[Callable[[Event], Any], Callable[[Event], Coroutine[Any, Any, Any]]]


class EventBus:
    """
    Thread-safe & async-compatible event bus.
    Allows decoupling subsystems while maintaining deterministic event handling.
    """

    def __init__(self):
        self._subscribers: Dict[str, List[HandlerType]] = {}
        self._global_subscribers: List[HandlerType] = []
        self._published_count: int = 0
        self._error_count: int = 0

    @property
    def published_count(self) -> int:
        return self._published_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def subscribe(self, event_type: str, handler: HandlerType) -> None:
        """Subscribe a callable (sync or async) to a specific event type or '*' for all."""
        if event_type == "*":
            if handler not in self._global_subscribers:
                self._global_subscribers.append(handler)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: HandlerType) -> bool:
        """Unsubscribe handler. Returns True if found and removed."""
        if event_type == "*":
            if handler in self._global_subscribers:
                self._global_subscribers.remove(handler)
                return True
        elif event_type in self._subscribers:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                return True
        return False

    def clear(self, event_type: Optional[str] = None) -> None:
        """Clear subscribers for a type or all types."""
        if event_type is None:
            self._subscribers.clear()
            self._global_subscribers.clear()
        elif event_type == "*":
            self._global_subscribers.clear()
        elif event_type in self._subscribers:
            self._subscribers[event_type].clear()

    def publish_sync(self, event: Event) -> None:
        """
        Publish an event synchronously to all registered sync handlers.
        If an async handler is encountered, schedules it if an event loop is running.
        """
        self._published_count += 1
        handlers = list(self._subscribers.get(event.event_type, [])) + list(self._global_subscribers)

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(handler(event))
                    except RuntimeError:
                        logger.warning(
                            f"Async handler {handler.__name__} for {event.event_type} "
                            f"cannot run without an active event loop."
                        )
                else:
                    handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Error executing event handler {handler} for {event.event_type}: {e}", exc_info=True)

    async def publish(self, event: Event) -> None:
        """
        Publish an event asynchronously. Awaits all async handlers and executes sync handlers.
        """
        self._published_count += 1
        handlers = list(self._subscribers.get(event.event_type, [])) + list(self._global_subscribers)

        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Async event handler {handler} failed for {event.event_type}: {e}", exc_info=True)


# Global default event bus instance
event_bus = EventBus()
