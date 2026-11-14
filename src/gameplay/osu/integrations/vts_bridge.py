"""
Stub module for VTS Bridge.
Pure gameplay mode: VTuber and Live2D integrations are disabled.
"""

from dataclasses import dataclass

@dataclass
class TargetHitEvent:
    hit_id: str = ""
    is_slider: bool = False
    accuracy: float = 100.0

@dataclass
class ComboMilestoneEvent:
    combo: int = 0
    milestone: int = 0

@dataclass
class ComboBreakEvent:
    broken_at_combo: int = 0

@dataclass
class SongCompletedEvent:
    beatmap_title: str = ""
    final_score: int = 0
    final_accuracy: float = 100.0

@dataclass
class SongFailedEvent:
    beatmap_title: str = ""
    failed_at_time_ms: float = 0.0


class OsuVTSBridge:
    def __init__(self, *args, **kwargs):
        self.enabled = False
        self.is_connected = False
        self.EMOTION_MAP = {}

    def start(self):
        pass

    def stop(self):
        pass

    def trigger_emotion(self, *args, **kwargs):
        pass

    def trigger_bop(self, *args, **kwargs):
        pass

    def update_gameplay_tracking(self, *args, **kwargs):
        pass

    def reset_all(self):
        pass


class OsuMotionBridge(OsuVTSBridge):
    pass


osu_vts_bridge = OsuVTSBridge()
