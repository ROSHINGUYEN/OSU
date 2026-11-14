"""
Central Configuration for Shiori AI osu! Gameplay System.
Provides type validation, default settings, and environment variable overrides.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaptureConfig:
    fps: int = 240
    device_idx: int = 0
    method: str = "memory"  # "memory" (Tosu zero-latency memory reader), "none", "mock"
    playfield_roi: Optional[List[int]] = None  # [x, y, w, h]


@dataclass
class TimingConfig:
    busy_wait_threshold_ms: float = 2.0
    default_system_latency_ms: float = 4.5
    learning_rate_adaptive_offset: float = 0.05
    max_adaptive_offset_ms: float = 50.0
    clock_drift_reset_threshold_ms: float = 20.0


@dataclass
class InputConfig:
    standard_keys: List[str] = field(default_factory=lambda: ["z", "x"])
    taiko_keys: Dict[str, str] = field(
        default_factory=lambda: {
            "don_left": "f",
            "don_right": "j",
            "katsu_left": "d",
            "katsu_right": "k",
        }
    )
    catch_keys: Dict[str, str] = field(
        default_factory=lambda: {"left": "left", "right": "right", "dash": "shift"}
    )
    mania_4k_keys: List[str] = field(default_factory=lambda: ["d", "f", "j", "k"])
    mania_7k_keys: List[str] = field(default_factory=lambda: ["s", "d", "f", "space", "j", "k", "l"])
    cursor_smoothing_steps: int = 5
    micro_jitter_enabled: bool = True  # Gaussian human-like jitter
    micro_jitter_sigma_ms: float = 1.2
    backend: str = "win32_sendinput"  # "win32_sendinput", "mock"


@dataclass
class SafetyConfig:
    lease_duration_ms: int = 50
    emergency_hotkey: str = "F12"
    max_stuck_key_duration_ms: int = 3000
    auto_recovery_enabled: bool = True
    max_recovery_retries: int = 3


@dataclass
class PlanningConfig:
    horizon_ms: float = 1000.0
    replan_interval_ms: float = 20.0
    w_hit: float = 1.0
    w_combo: float = 1.5
    w_acc: float = 0.8
    w_risk: float = 2.0
    w_cost: float = 0.5


@dataclass
class OsuGameplayConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    input: InputConfig = field(default_factory=InputConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    log_level: str = "INFO"
    telemetry_dir: str = "data/osu_telemetry"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OsuGameplayConfig":
        config = cls()
        if "capture" in data:
            for k, v in data["capture"].items():
                if hasattr(config.capture, k):
                    setattr(config.capture, k, v)
        if "timing" in data:
            for k, v in data["timing"].items():
                if hasattr(config.timing, k):
                    setattr(config.timing, k, v)
        if "input" in data:
            for k, v in data["input"].items():
                if hasattr(config.input, k):
                    setattr(config.input, k, v)
        if "safety" in data:
            for k, v in data["safety"].items():
                if hasattr(config.safety, k):
                    setattr(config.safety, k, v)
        if "planning" in data:
            for k, v in data["planning"].items():
                if hasattr(config.planning, k):
                    setattr(config.planning, k, v)
        if "log_level" in data:
            config.log_level = data["log_level"]
        if "telemetry_dir" in data:
            config.telemetry_dir = data["telemetry_dir"]
        return config


# Global default configuration instance
default_config = OsuGameplayConfig()
