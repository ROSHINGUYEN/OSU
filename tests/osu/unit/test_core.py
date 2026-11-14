"""
Unit Tests for Shiori AI osu! Gameplay Core Foundation.
Tests Clock, EventBus, DataChannels, StateStore, and Configuration.
"""

import asyncio
import threading
import time
import pytest

from src.gameplay.osu.core import (
    HighResolutionClock,
    Event,
    EventBus,
    DataChannel,
    CentralStateStore,
    OperatingState,
    GameMode,
    ModeDetectionStatus,
    RecoveryFsmState,
    OsuGameplayConfig,
    default_config,
)
from src.gameplay.osu.core.state_store import SystemState, WorldState


# ==========================================
# 1. Clock Tests
# ==========================================

def test_high_resolution_clock_monotonic():
    clk = HighResolutionClock()
    t1 = clk.now_ns()
    time.sleep(0.005)
    t2 = clk.now_ns()

    assert t2 > t1
    elapsed_ms = clk.elapsed_since_ms(t1)
    assert 3.5 <= elapsed_ms <= 150.0


def test_clock_offset_calibration():
    clk = HighResolutionClock()
    base_ns = clk.now_ns()
    clk.set_offset_ns(10_000_000)  # +10ms
    assert clk.offset_ns == 10_000_000
    assert clk.offset_ms == 10.0

    clk.adjust_offset_ns(-2_000_000)  # -2ms -> 8ms
    assert clk.offset_ns == 8_000_000
    assert clk.offset_ms == 8.0


def test_clock_sleep_until():
    clk = HighResolutionClock()
    start_ns = clk.now_ns()
    target_ns = start_ns + 5_000_000  # 5ms

    overshoot_ns = clk.sleep_until_ns(target_ns)
    now_ns = clk.now_ns()

    assert now_ns >= target_ns
    assert overshoot_ns >= 0
    # Overshoot should typically be small (< 2ms on Windows)
    assert overshoot_ns < 4_000_000


# ==========================================
# 2. Event Bus Tests
# ==========================================

def test_event_bus_sync_dispatch():
    bus = EventBus()
    received = []

    def on_frame(event: Event):
        received.append(event)

    bus.subscribe("frame_captured", on_frame)
    ev = Event(event_type="frame_captured", payload={"fps": 120})
    bus.publish_sync(ev)

    assert len(received) == 1
    assert received[0].payload["fps"] == 120
    assert bus.published_count == 1
    assert bus.error_count == 0


def test_event_bus_wildcard_and_unsubscribe():
    bus = EventBus()
    wildcard_events = []
    specific_events = []

    def on_all(event: Event):
        wildcard_events.append(event)

    def on_hit(event: Event):
        specific_events.append(event)

    bus.subscribe("*", on_all)
    bus.subscribe("hit_observed", on_hit)

    bus.publish_sync(Event(event_type="hit_observed", payload={"result": "300"}))
    bus.publish_sync(Event(event_type="mode_changed", payload={"mode": "taiko"}))

    assert len(wildcard_events) == 2
    assert len(specific_events) == 1

    # Unsubscribe specific
    assert bus.unsubscribe("hit_observed", on_hit) is True
    bus.publish_sync(Event(event_type="hit_observed", payload={"result": "100"}))

    assert len(specific_events) == 1
    assert len(wildcard_events) == 3


def test_event_bus_handler_exception_containment():
    bus = EventBus()

    def faulty_handler(event: Event):
        raise RuntimeError("Crash in subscriber")

    bus.subscribe("test_error", faulty_handler)
    # Should not raise exception out of publish_sync
    bus.publish_sync(Event(event_type="test_error"))

    assert bus.error_count == 1
    assert bus.published_count == 1


@pytest.mark.asyncio
async def test_event_bus_async_dispatch():
    bus = EventBus()
    async_received = []

    async def async_handler(event: Event):
        await asyncio.sleep(0.001)
        async_received.append(event.event_type)

    bus.subscribe("async_event", async_handler)
    await bus.publish(Event(event_type="async_event"))

    assert async_received == ["async_event"]


# ==========================================
# 3. Data Channels Tests
# ==========================================

def test_data_channel_ring_buffer_capacity():
    channel: DataChannel[int] = DataChannel("test_channel", capacity=3)
    assert len(channel) == 0

    channel.push(1)
    channel.push(2)
    channel.push(3)
    assert len(channel) == 3
    assert channel.get_latest() == 3
    assert channel.dropped_count == 0

    # Overwrite oldest (drops 1)
    channel.push(4)
    assert len(channel) == 3
    assert channel.get_latest() == 4
    assert channel.dropped_count == 1
    assert channel.get_recent(3) == [2, 3, 4]


def test_data_channel_thread_safety():
    channel: DataChannel[int] = DataChannel("threaded_channel", capacity=100)

    def worker(start_val: int):
        for i in range(50):
            channel.push(start_val + i)

    t1 = threading.Thread(target=worker, args=(0,))
    t2 = threading.Thread(target=worker, args=(1000,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert channel.total_pushed == 100
    assert len(channel) == 100


# ==========================================
# 4. Central State Store Tests
# ==========================================

def test_state_store_initial_domains():
    store = CentralStateStore()
    assert store.version == 0
    assert store.system.state == OperatingState.INITIALIZING
    assert store.mode.mode == GameMode.UNKNOWN
    assert store.beatmap.is_loaded is False
    assert store.world.current_combo == 0
    assert store.timing.adaptive_offset_ms == 0.0
    assert store.recovery.state == RecoveryFsmState.IDLE


def test_state_store_authorized_mutation():
    store = CentralStateStore()
    notified = []

    def on_world_change(new_state):
        notified.append(new_state)

    store.subscribe("world", on_world_change)

    # Mutate from allowed caller 'world_service'
    new_world = store.mutate_domain(
        "world",
        lambda curr: WorldState(current_combo=42, accuracy=99.5, state_version=curr.state_version + 1),
        caller="world_service",
    )

    assert store.version == 1
    assert store.world.current_combo == 42
    assert store.world.accuracy == 99.5
    assert len(notified) == 1
    assert notified[0].current_combo == 42


def test_state_store_unauthorized_mutation_rejected():
    store = CentralStateStore()

    with pytest.raises(PermissionError):
        # 'hacker' is not allowed to mutate 'safety'
        store.mutate_domain(
            "safety",
            lambda curr: curr,
            caller="unauthorized_caller",
        )

    assert store.version == 0


def test_state_store_unknown_domain_rejected():
    store = CentralStateStore()

    with pytest.raises(ValueError):
        store.mutate_domain(
            "nonexistent_domain",
            lambda curr: curr,
            caller="system",
        )


# ==========================================
# 5. Config Tests
# ==========================================

def test_default_config_integrity():
    cfg = default_config
    assert cfg.capture.fps == 240
    assert cfg.capture.method == "memory"
    assert cfg.timing.busy_wait_threshold_ms == 2.0
    assert cfg.input.standard_keys == ["z", "x"]
    assert cfg.safety.emergency_hotkey == "F12"
    assert cfg.planning.horizon_ms == 1000.0


def test_config_from_dict_override():
    custom = {
        "capture": {"fps": 240, "method": "mock"},
        "timing": {"default_system_latency_ms": 6.2},
        "safety": {"emergency_hotkey": "ESC"},
        "log_level": "DEBUG",
    }
    cfg = OsuGameplayConfig.from_dict(custom)

    assert cfg.capture.fps == 240
    assert cfg.capture.method == "mock"
    assert cfg.timing.default_system_latency_ms == 6.2
    assert cfg.safety.emergency_hotkey == "ESC"
    assert cfg.log_level == "DEBUG"
    # Unmodified fields should keep defaults
    assert cfg.input.standard_keys == ["z", "x"]
