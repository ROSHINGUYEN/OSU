"""
Main Application Entrypoint and Lifecycle Orchestrator for Shiori AI osu! Gameplay.
Connects Perception, World State, Planning, Input, Safety, and Evaluation domains into a single runtime loop.
"""

import ctypes
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.gameplay.osu.beatmap.loader import BeatmapLoader
from src.gameplay.osu.beatmap.models import (
    BeatmapData,
    HitCircle,
    Slider,
    Spinner,
)
from src.gameplay.osu.beatmap.timeline import HitObjectTimeline
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.data_channels import frame_channel
from src.gameplay.osu.core.gpu_engine import gpu_engine
from src.gameplay.osu.core.state_store import (
    GameMode,
    ModeDetectionStatus,
    OperatingState,
    SystemState,
    state_store,
)
from src.gameplay.osu.evaluation.feedback import (
    FeedbackEvaluator,
    feedback_evaluator,
)
from src.gameplay.osu.evaluation.game_memory import osu_game_memory
from src.gameplay.osu.evaluation.telemetry import (
    TelemetryLogger,
    telemetry_logger,
)
from src.gameplay.osu.input.action_model import (
    ActionType,
    CursorOwnerState,
    InputStateStatus,
)
from src.gameplay.osu.input.controller import (
    CentralInputController,
    input_controller,
)
from src.gameplay.osu.input.queue import CentralInputQueue, input_queue
from src.gameplay.osu.integrations.discord_reporter import osu_discord_reporter
from src.gameplay.osu.integrations.tosu_client import tosu_client
from src.gameplay.osu.integrations.vts_bridge import osu_vts_bridge
import src.gameplay.osu.perception.capture as osu_capture
from src.gameplay.osu.perception.capture import (
    CaptureService,
    capture_service,
)
from src.gameplay.osu.perception.detector import VisionDetector
from src.gameplay.osu.perception.mode_detector import (
    ModeDetectionService,
    mode_detection_service,
)
from src.gameplay.osu.perception.preprocessor import (
    PlayfieldPreprocessor,
)
from src.gameplay.osu.perception.tracker import ObjectTracker
from src.gameplay.osu.planning.planner import (
    RollingHorizonPlanner,
    rolling_planner,
)
from src.gameplay.osu.safety.supervisor import (
    SafetySupervisor,
    safety_supervisor,
)
from src.gameplay.osu.timing.calibration import timing_calibrator
from src.gameplay.osu.world.reconciliation import (
    ReconciliationService,
    reconciliation_service,
)
from src.gameplay.osu.world.world_state import (
    WorldStateManager,
    world_state_manager,
)

logger = logging.getLogger("Shiori.Osu.App")


class OsuGameplayApp:
    """Master coordinator executing real-time game play loops and managing session lifecycle."""

    def __init__(
        self,
        capture: CaptureService = capture_service,
        reconciler: ReconciliationService = reconciliation_service,
        planner: RollingHorizonPlanner = rolling_planner,
        queue: CentralInputQueue = input_queue,
        controller: CentralInputController = input_controller,
        safety: SafetySupervisor = safety_supervisor,
        feedback: FeedbackEvaluator = feedback_evaluator,
        telemetry: TelemetryLogger = telemetry_logger,
        preprocessor: PlayfieldPreprocessor = None,
        world_manager: WorldStateManager = world_state_manager,
    ):
        self.capture = capture
        self.reconciler = reconciler
        self.planner = planner
        self.queue = queue
        self.controller = controller
        self.safety = safety
        self.feedback = feedback
        self.telemetry = telemetry
        self.preprocessor = preprocessor or PlayfieldPreprocessor()
        self.world_manager = world_manager
        self.detector = VisionDetector(self.preprocessor.bounds)
        self.tracker = ObjectTracker()

        # Propagate unified preprocessor instance to planner and all mode brains
        if hasattr(self.planner, "set_preprocessor"):
            self.planner.set_preprocessor(self.preprocessor)
        elif hasattr(self.planner, "_brains"):
            for brain in self.planner._brains.values():
                if hasattr(brain, "preprocessor"):
                    brain.preprocessor = self.preprocessor

        self._active_beatmap: Optional[BeatmapData] = None
        self._timeline: Optional[HitObjectTimeline] = None
        self._session_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._timeline_anchor_ns: int = 0
        self._is_running = False

        self._current_emotion: str = "neutral"
        self._last_thought: str = ""
        self._last_combo_milestone: int = 0
        self._last_reported_combo: int = 0
        self._last_thought_time_s: float = 0.0
        self._last_action_desc: Optional[str] = None
        self._actions_dispatched_total: int = 0
        self._last_optical_hit_ms: float = 0.0
        self._last_optical_don_hand: str = "left"
        self._last_optical_katsu_hand: str = "left"
        self._last_hit_by_key: Dict[str, float] = {"z": 0.0, "x": 0.0}
        self._last_alarm_time: float = 0.0
        self._last_key_press_time_ms: Dict[str, float] = {}

        # Tosu anchor sync: track last known tosu time to detect song restart
        self._last_tosu_time_ms: float = -1.0
        self._tosu_anchor_synced: bool = False
        self._last_processed_frame_seq: int = -1
        self._auto_aim_enabled: bool = True
        self._active_beatmap_id: int = -1
        self.enable_vtuber_bridge: bool = True
        self._session_id: str = "session_01"
        self._current_session_start_time: float = 0.0
        self._session_result_logged: bool = False
        self._reported_fail: bool = False
        self._reported_pass: bool = False
        self._was_active_play: bool = False
        self._last_notified_hard_map_id: int = -1
        self._last_step_miss_count: int = 0

        # osu!lazer pause and resume tracking
        self._lazer_is_paused: bool = False
        self._pause_start_ns: int = 0
        self._lazer_resumed: bool = False
        self._current_lazer_player_screen: str = ""
        self._last_seen_player_screen: str = ""
        self._cached_runtime_log_path: Optional[Path] = None
        self._last_runtime_log_scan: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def auto_aim_enabled(self) -> bool:
        return getattr(self, "_auto_aim_enabled", True)

    @auto_aim_enabled.setter
    def auto_aim_enabled(self, val: bool) -> None:
        self._auto_aim_enabled = bool(val)
        if hasattr(self, "controller"):
            self.controller.auto_aim_enabled = bool(val)
        if hasattr(self, "planner"):
            std_brain = self.planner.get_brain(GameMode.STANDARD)
            if hasattr(std_brain, "auto_aim_enabled"):
                std_brain.auto_aim_enabled = bool(val)

    def _handle_song_completed(self) -> None:
        """Called immediately when a song finishes successfully (enters Results or SongSelect)."""
        acc = self.feedback.metrics.accuracy
        max_combo = self.feedback.metrics.max_combo
        total_hits = self.feedback.metrics.total_hits
        logger.info(f"[OsuGameplayApp] Match completed! Acc: {acc:.2f}%, Max Combo: {max_combo}x, Hits: {total_hits}")

        self._log_session_to_memory(result="pass", failure_reason="song_completed")

        commentary = (
            f"Hoan hô Ba Roshi ơi! Con gái vừa hoàn thành xuất sắc bài hát này rồi nè! "
            f"Độ chính xác đạt {acc:.1f}%, combo cao nhất {max_combo}x luôn nha! Ba thấy Shiori giỏi hơm? ♡"
        )
        self._emit_emotion_and_thought(
            emotion="happy",
            intensity=1.0,
            duration=5.0,
            text=commentary,
            reason="song_completed",
        )

        try:
            b_meta = getattr(self._active_beatmap, "metadata", None)
            title = state_store.beatmap.title or (getattr(b_meta, "title", "osu! Session") if b_meta else "osu! Session")
            artist = state_store.beatmap.artist or (getattr(b_meta, "artist", "Various Artists") if b_meta else "Various Artists")
            version = getattr(b_meta, "version", "Normal") if b_meta else "Normal"
            osu_discord_reporter.send_game_summary(
                title=title,
                artist=artist,
                difficulty=version,
                mode=state_store.mode.mode.value,
                accuracy=acc,
                combo=self.feedback.metrics.current_combo,
                max_combo=max_combo,
                hits=dict(self.feedback.metrics.hits),
                commentary=commentary,
                is_failed=False,
                status="completed",
                lead_time_ms=getattr(timing_calibrator, "system_latency_ms", 12.0),
                emotion="happy",
            )
        except Exception as e:
            logger.debug(f"Failed to dispatch pass summary to Discord: {e}")

    def _emit_emotion_and_thought(
        self,
        emotion: str = "neutral",
        intensity: float = 1.0,
        duration: float = 3.0,
        text: str = "",
        reason: str = "gameplay",
    ) -> None:
        """Emit emotional expression to 2D model motion system and log internal thought."""
        self._current_emotion = emotion
        self._last_thought = text
        return

    @staticmethod
    def get_lazer_ruleset_mode() -> Optional[GameMode]:
        """
        Pure osu! Standard dedicated ruleset.
        Always operates in osu! Standard mode.
        """
        return GameMode.STANDARD

    def load_beatmap_from_string(self, content: str, key: str = "in_memory") -> bool:
        """Parse and prepare beatmap."""
        try:
            self._active_beatmap, self._timeline = BeatmapLoader.load_from_string(content, key=key)
            BeatmapLoader.sync_to_state_store(self._active_beatmap)
            # Shiori is 100% dedicated to osu! Standard mode; preserve non-standard beatmap mode if explicitly defined
            if state_store.mode.user_overridden:
                target_mode = state_store.mode.mode
            elif self._active_beatmap.mode != GameMode.STANDARD:
                target_mode = self._active_beatmap.mode
            else:
                target_mode = GameMode.STANDARD
            mode_detection_service.evaluate_mode(beatmap_mode=target_mode)
            if hasattr(self._active_beatmap, "difficulty"):
                std_brain = self.planner.get_brain(GameMode.STANDARD)
                if hasattr(self._active_beatmap.difficulty, "overall_difficulty"):
                    od_val = float(self._active_beatmap.difficulty.overall_difficulty)
                    if hasattr(std_brain, "od"):
                        std_brain.od = od_val
                        logger.info(f"Synchronized StandardBrain OD to beatmap difficulty: OD={od_val}")
                if hasattr(self._active_beatmap.difficulty, "circle_size"):
                    cs_val = float(self._active_beatmap.difficulty.circle_size)
                    if hasattr(std_brain, "cs"):
                        std_brain.cs = cs_val
                        logger.info(f"Synchronized StandardBrain CS to beatmap difficulty: CS={cs_val}")
                    if hasattr(self, "preprocessor") and hasattr(self.preprocessor, "set_beatmap_difficulty"):
                        self.preprocessor.set_beatmap_difficulty(
                            self._active_beatmap.difficulty,
                            getattr(self._active_beatmap, "general", None),
                        )
                    if hasattr(self, "detector") and hasattr(self.detector, "update_circle_size"):
                        self.detector.update_circle_size(cs_val, getattr(self.preprocessor.bounds, "scale", 1.875))
                    if hasattr(self, "reconciler") and hasattr(self.reconciler, "spatial_match_threshold"):
                        r_screen = self.preprocessor.get_hit_circle_radius(cs_val)
                        self.reconciler.spatial_match_threshold = max(35.0, min(120.0, r_screen * 1.15))
            logger.info(f"Loaded beatmap: {self._active_beatmap.metadata.title} (Mode: {target_mode.value})")
            self._on_beatmap_activated(target_mode)
            return True
        except Exception as e:
            logger.error(f"Failed to load beatmap: {e}", exc_info=True)
            return False

    def load_beatmap_file(self, file_path: str) -> bool:
        """Load beatmap from disk."""
        try:
            self._active_beatmap, self._timeline = BeatmapLoader.load_from_file(file_path)
            BeatmapLoader.sync_to_state_store(self._active_beatmap)
            # Shiori is 100% dedicated to osu! Standard mode; preserve non-standard beatmap mode if explicitly defined
            if state_store.mode.user_overridden:
                target_mode = state_store.mode.mode
            elif self._active_beatmap.mode != GameMode.STANDARD:
                target_mode = self._active_beatmap.mode
            else:
                target_mode = GameMode.STANDARD
            mode_detection_service.evaluate_mode(beatmap_mode=target_mode)
            if hasattr(self._active_beatmap, "difficulty"):
                std_brain = self.planner.get_brain(GameMode.STANDARD)
                if hasattr(self._active_beatmap.difficulty, "overall_difficulty"):
                    od_val = float(self._active_beatmap.difficulty.overall_difficulty)
                    if hasattr(std_brain, "od"):
                        std_brain.od = od_val
                        logger.info(f"Synchronized StandardBrain OD to beatmap difficulty: OD={od_val}")
                if hasattr(self._active_beatmap.difficulty, "circle_size"):
                    cs_val = float(self._active_beatmap.difficulty.circle_size)
                    if hasattr(std_brain, "cs"):
                        std_brain.cs = cs_val
                        logger.info(f"Synchronized StandardBrain CS to beatmap difficulty: CS={cs_val}")
                    if hasattr(self, "preprocessor") and hasattr(self.preprocessor, "set_beatmap_difficulty"):
                        self.preprocessor.set_beatmap_difficulty(
                            self._active_beatmap.difficulty,
                            getattr(self._active_beatmap, "general", None),
                        )
                    if hasattr(self, "detector") and hasattr(self.detector, "update_circle_size"):
                        self.detector.update_circle_size(cs_val, getattr(self.preprocessor.bounds, "scale", 1.875))
                    if hasattr(self, "reconciler") and hasattr(self.reconciler, "spatial_match_threshold"):
                        r_screen = self.preprocessor.get_hit_circle_radius(cs_val)
                        self.reconciler.spatial_match_threshold = max(35.0, min(120.0, r_screen * 1.15))
            logger.info(f"Loaded beatmap: {self._active_beatmap.metadata.title} (Mode: {target_mode.value})")
            self._on_beatmap_activated(target_mode)
            return True
        except Exception as e:
            logger.error(f"Failed to load beatmap file '{file_path}': {e}", exc_info=True)
            return False

    def _log_session_to_memory(self, result: str = "pass", failure_reason: str = "normal_exit") -> None:
        """Write session summary to persistent Game Memory JSONL."""
        if getattr(self, "_session_result_logged", False):
            return
        self._session_result_logged = True

        try:
            b_meta = getattr(self._active_beatmap, "metadata", None)
            title = state_store.beatmap.title or (getattr(b_meta, "title", "osu! Session") if b_meta else "osu! Session")
            artist = state_store.beatmap.artist or (getattr(b_meta, "artist", "Various Artists") if b_meta else "Various Artists")
            version = getattr(b_meta, "version", "Normal") if b_meta else "Normal"
            bm_id = int(getattr(b_meta, "beatmap_id", 0) or getattr(tosu_client, "beatmap_id", 0) or 0)
            mode_str = state_store.mode.mode.value if state_store.mode.mode != GameMode.UNKNOWN else "standard"

            # Extract metrics with Tosu memory priority if connected
            tosu_hits_count = (tosu_client.hits_300 + tosu_client.hits_100 + tosu_client.hits_50 + tosu_client.hits_miss) if tosu_client.is_connected else 0
            if tosu_client.is_connected and tosu_hits_count > 0:
                acc_val = tosu_client.accuracy
                combo_val = tosu_client.current_combo
                max_combo_val = tosu_client.max_combo
                miss_val = tosu_client.hits_miss
                hits_val = {
                    "300": tosu_client.hits_300,
                    "100": tosu_client.hits_100,
                    "50": tosu_client.hits_50,
                    "miss": tosu_client.hits_miss,
                }
            else:
                acc_val = self.feedback.metrics.accuracy
                combo_val = self.feedback.metrics.current_combo
                max_combo_val = self.feedback.metrics.max_combo
                miss_val = self.feedback.metrics.count_miss
                hits_val = dict(self.feedback.metrics.hits if hasattr(self.feedback.metrics, "hits") else {"300": self.feedback.metrics.count_300, "100": self.feedback.metrics.count_100, "50": self.feedback.metrics.count_50, "miss": self.feedback.metrics.count_miss})

            timing_errors = list(self.feedback.metrics.timing_errors_ms)
            avg_err = self.feedback.metrics.timing_error_mean
            variance_err = self.feedback.metrics.timing_error_variance
            provenance_val = self.feedback.provenance.value if hasattr(self.feedback, "provenance") else "observed"

            tuning_dict = {
                "system_latency_ms": timing_calibrator.system_latency_ms,
                "baseline_hardware_offset_ms": timing_calibrator.baseline_hardware_offset_ms,
                "adaptive_offset_ms": timing_calibrator.adaptive_offset_ms,
                "total_lead_time_ms": timing_calibrator.total_lead_time_ms,
                "stale_actions_rejected": self.queue.stale_rejected_count,
            }

            duration_s = max(0.1, time.time() - getattr(self, "_current_session_start_time", time.time()))
            is_stream_heavy = False
            if self._active_beatmap and getattr(self._active_beatmap, "hit_objects", None):
                hos = self._active_beatmap.hit_objects
                short_gaps = sum(1 for i in range(1, len(hos)) if 0.0 < (hos[i].time_ms - hos[i - 1].time_ms) <= 115.0)
                is_stream_heavy = len(hos) >= 20 and (short_gaps / max(1, len(hos) - 1)) >= 0.25

            osu_game_memory.log_session({
                "session_id": getattr(self, "_session_id", "session_01"),
                "beatmap_id": bm_id,
                "beatmap_title": title,
                "artist": artist,
                "version": version,
                "mode": mode_str,
                "result": result,
                "accuracy": acc_val,
                "combo": combo_val,
                "max_combo": max_combo_val,
                "miss_count": miss_val,
                "hits": hits_val,
                "avg_timing_error_ms": avg_err,
                "timing_error_variance": variance_err,
                "provenance": provenance_val,
                "timing_errors": timing_errors,
                "failure_reason": failure_reason,
                "duration_s": duration_s,
                "is_stream_heavy": is_stream_heavy,
                "tuning_applied": tuning_dict,
            })
        except Exception as e:
            logger.warning(f"Could not log session to game memory: {e}")

    def _on_beatmap_activated(self, target_mode: GameMode) -> None:
        """Trigger game memory self-tuning and detect fast/difficult beatmaps."""
        if not self._active_beatmap:
            return

        # Neutralize any residual inputs, stale deadman cooldowns, or stale cursor positions
        try:
            self.controller.emergency_release_all()
        except Exception:
            pass

        # Reset timing calibrator to clean baseline on every song entry & tune for BPM
        try:
            timing_calibrator.reset_offset()
            is_mem = tosu_client.is_connected or getattr(self, "engine_mode", "") == "memory"
            bpm_val = float(getattr(self._active_beatmap, "bpm", 0.0) or 0.0)
            if bpm_val <= 0.0:
                tps = getattr(self._active_beatmap, "timing_points", [])
                uninh = [p for p in tps if getattr(p, "uninherited", True)]
                if uninh and getattr(uninh[0], "beat_length", 0.0) > 0.0:
                    bpm_val = 60000.0 / uninh[0].beat_length
                else:
                    bpm_val = 160.0
            timing_calibrator.calibrate_for_bpm(bpm_val, is_memory_mode=is_mem)
            if is_mem:
                # Hardware SendInput + Windows CSRSS + SDL2 polling has a 14-16ms latency baseline.
                # Initialize hardware baseline offset so initial hits land on 0.0ms before closed-loop feedback tunes it.
                init_offset = max(0.0, 14.5 - timing_calibrator.system_latency_ms)
                timing_calibrator.set_baseline_hardware_offset_ms(init_offset)
        except Exception:
            pass

        # Refresh osu! game window coordinates & resolution immediately on song activation
        try:
            if hasattr(self, "controller") and hasattr(self.controller, "backend"):
                if not self.controller.backend.__class__.__name__.startswith("Mock"):
                    reg = osu_capture.find_osu_window_region()
                    if reg:
                        self.capture.update_region(reg)
                        ol, ot, oright, obottom = reg
                        ow = oright - ol
                        oh = obottom - ot
                        if ow > 200 and oh > 200:
                            self.preprocessor.update_window_position(ol, ot)
                            self.preprocessor.update_resolution(ow, oh)
                            self.detector.update_bounds(self.preprocessor.bounds)
                            cs_active = getattr(self.preprocessor, "active_circle_size", 4.0)
                            self.detector.update_circle_size(cs_active, self.preprocessor.bounds.scale)
                            if hasattr(self, "reconciler") and hasattr(self.reconciler, "spatial_match_threshold"):
                                r_screen = self.preprocessor.get_hit_circle_radius(cs_active)
                                self.reconciler.spatial_match_threshold = max(35.0, min(120.0, r_screen * 1.15))
                            logger.info(
                                f"Refreshed resolution for active beatmap: {ow}x{oh} at ({ol}, {ot}) | "
                                f"CS={cs_active:.1f} (R_screen={self.preprocessor.get_hit_circle_radius(cs_active):.1f}px)"
                            )
        except Exception:
            pass

        self._pre_aimed_first_note = False
        bm_meta = getattr(self._active_beatmap, "metadata", None)
        bm_id_val = int(getattr(bm_meta, "beatmap_id", 0) or getattr(tosu_client, "beatmap_id", 0) or 0)
        bm_title_val = str(getattr(bm_meta, "title", "") or getattr(tosu_client, "beatmap_title", "") or "")
        mode_val = target_mode.value

        # 1. Apply Game Memory self-tuning & reflection
        try:
            tuning = osu_game_memory.apply_self_tuning(beatmap_id=bm_id_val, mode=mode_val, title=bm_title_val)
            from src.gameplay.osu.planning.adaptive_policy import adaptive_policy
            std_brain = self.planner.get_brain(GameMode.STANDARD)
            if std_brain and hasattr(std_brain, "smoothing_steps"):
                std_brain.smoothing_steps = max(2, int(adaptive_policy.get_param("aim_smoothing_steps", 4)))
            insight = tuning.get("shiori_insight", "")
            if insight:
                self._emit_emotion_and_thought(
                    emotion="thinking" if tuning.get("is_difficult") else "happy",
                    intensity=1.0,
                    duration=3.5,
                    text=insight,
                    reason="memory_reflection",
                )
        except Exception as e:
            logger.debug(f"Game memory tuning lookup failed: {e}")

        # 2. Check if beatmap is fast & hard -> trigger thinking emotion
        try:
            bpm = float(getattr(self._active_beatmap, "bpm", 0.0) or 0.0)
            objs_count = len(self._timeline.all_objects) if self._timeline and hasattr(self._timeline, "all_objects") else 0
            duration_s = max(1.0, (float(getattr(self._active_beatmap, "total_length_ms", 60000.0) or 60000.0) / 1000.0))
            density = objs_count / duration_s
            if (bpm >= 200.0 or density >= 5.5) and getattr(self, "_last_notified_hard_map_id", -1) != bm_id_val:
                self._last_notified_hard_map_id = bm_id_val
                thought = f"Bàn này tiết tấu nhanh và dồn dập ghê nè Ba ({bpm:.0f} BPM, {density:.1f} nốt/s)! Shiori đang tập trung cao độ phân tích timeline nhịp điệu nè! 💭"
                self._emit_emotion_and_thought(
                    emotion="thinking",
                    intensity=1.0,
                    duration=3.5,
                    text=thought,
                    reason="hard_map_alert",
                )
        except Exception as e:
            logger.debug(f"Hard map check failed: {e}")

        # 3. Match start announcement on Discord suppressed per user request (only final summary on finish)

    def _check_lazer_gameplay_state(self, now_s: float) -> Tuple[bool, Optional[str]]:
        """
        Query osu!lazer runtime log to detect:
        1. Whether player is currently in gameplay (PlayerLoader / SoloPlayer / MultiplayerPlayer)
        2. Active working beatmap title string
        3. Real-time gameplay clock state (playing vs paused)
        Throttled to run at most every 0.05s to keep overhead minimal (<0.5ms).
        """
        cached_result = getattr(self, "_cached_lazer_play_state", (False, None))
        if (now_s - getattr(self, "_last_lazer_state_check_time", 0.0)) < 0.05:
            return cached_result

        self._last_lazer_state_check_time = now_s
        try:
            appdata = os.environ.get("APPDATA", "")
            if not appdata:
                return cached_result
            logs_dir = Path(appdata) / "osu" / "logs"
            if not logs_dir.exists():
                return cached_result

            # Fast cache: only re-scan directory every 4.0s or if cached log file is missing
            runtime_log = getattr(self, "_cached_runtime_log_path", None)
            if runtime_log is None or not runtime_log.exists() or (now_s - getattr(self, "_last_runtime_log_scan", 0.0)) > 4.0:
                self._last_runtime_log_scan = now_s
                runtime_logs = sorted(logs_dir.glob("*.runtime.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                if not runtime_logs:
                    return cached_result
                runtime_log = runtime_logs[0]
                self._cached_runtime_log_path = runtime_log

            with open(runtime_log, "r", encoding="utf-8", errors="ignore") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536))
                tail = f.read()

            # Find active screen
            screen_matches = list(re.finditer(r"entered\s+([A-Za-z0-9_#+]+)", tail))
            is_playing = False
            player_screen = ""
            is_in_player = False
            last_screen_idx = -1
            if screen_matches:
                last_match = screen_matches[-1]
                last_screen = last_match.group(1)
                last_screen_idx = last_match.start()
                # Gameplay screens in lazer: SoloPlayer, MultiplayerPlayer (PlayerLoader is pre-game loading screen)
                play_markers = ("Player",)
                menu_markers = ("SongSelect", "PlayerLoader", "MainMenu", "Results", "FailOverlay", "Prompt", "IntroTriangles")
                if any(m in last_screen for m in play_markers) and not any(m in last_screen for m in menu_markers):
                    is_in_player = True
                    player_screen = last_screen

            # Clock status within tail (pause/stop vs start)
            clock_starts = [m.start() for m in re.finditer(r"StartGameplayClock", tail)]
            clock_stops = [m.start() for m in re.finditer(r"StopGameplayClock", tail)]
            last_start = clock_starts[-1] if clock_starts else -1
            last_stop = clock_stops[-1] if clock_stops else -1
            self._last_lazer_clock_start_idx = last_start

            if is_in_player:
                # StopGameplayClock is only valid if it occurred AFTER entering the current player screen
                stop_in_current_screen = (last_stop > last_screen_idx) if last_screen_idx != -1 else (last_stop != -1)
                if last_stop > last_start and stop_in_current_screen:
                    # Clock is stopped while in player screen -> Paused!
                    # Live memory verification: If Tosu client is connected and its audio clock has advanced
                    # within the last 1.0s, the game is physically playing! A historical StopGameplayClock in
                    # runtime.log is stale from a prior pause/seek and must be ignored.
                    tosu_is_clock_advancing = (
                        tosu_client.is_connected
                        and tosu_client.is_playing
                        and (clock.now_ns() - getattr(self, "_last_tosu_clock_advancing_ns", 0)) < 1_000_000_000
                    )
                    if not tosu_is_clock_advancing:
                        is_playing = False
                        if not getattr(self, "_lazer_is_paused", False):
                            self._lazer_is_paused = True
                            self._pause_start_ns = clock.now_ns()
                            try:
                                self.controller.emergency_release_all()
                            except Exception:
                                pass
                            logger.info("osu!lazer gameplay PAUSE detected (StopGameplayClock) -> Entering pause standby.")
                    else:
                        is_playing = True
                        if getattr(self, "_lazer_is_paused", False):
                            self._lazer_is_paused = False
                            self._lazer_resumed = True
                            self._tosu_anchor_synced = False
                            logger.info("osu!lazer gameplay RESUME detected (in-game audio clock advancing)!")
                else:
                    # Clock is running in player screen -> Playing!
                    is_playing = True
                    if getattr(self, "_lazer_is_paused", False):
                        self._lazer_is_paused = False
                        self._lazer_resumed = True
                        self._tosu_anchor_synced = False
                        logger.info("osu!lazer gameplay RESUME detected (StartGameplayClock)!")
            else:
                # Outside player screen (e.g. SongSelect, Results, MainMenu)
                if getattr(self, "_lazer_is_paused", False):
                    self._lazer_is_paused = False
                    self._lazer_resumed = False
                    self._pause_start_ns = 0

            # Lead-in seek time
            # Prioritize seeks that occurred in the current player screen to prevent picking up ancient history
            player_tail = tail[last_screen_idx:] if last_screen_idx != -1 else tail
            seeks = re.findall(r"GameplayClockContainer seeking to (-?\d+)", player_tail)
            if not seeks:
                seeks = re.findall(r"GameplayClockContainer seeking to (-?\d+)", tail)
            leadin_ms = float(seeks[-1]) if seeks else 0.0

            # Find working beatmap
            maps = re.findall(r"Game-wide working beatmap updated to (.*)", tail)
            working_map = maps[-1].strip() if maps else None

            # Query ruleset mode directly from %APPDATA%\osu\game.ini (instant and authoritative for lazer)
            lazer_ruleset_mode = self.get_lazer_ruleset_mode()
            self._current_lazer_ruleset_mode = lazer_ruleset_mode

            self._current_lazer_player_screen = player_screen
            self._current_lazer_leadin_ms = leadin_ms
            res = (is_playing, working_map)
            self._cached_lazer_play_state = res
            return res
        except Exception:
            return cached_result

    def _try_auto_discover_beatmap(self) -> bool:
        """Attempt to auto-load the active beatmap by matching osu!lazer runtime log, window title, or cache."""
        is_live_backend = not self.controller.backend.__class__.__name__.startswith("Mock")
        if not is_live_backend:
            return False
        try:
            appdata = os.environ.get("APPDATA", "")
            if not appdata:
                return False
            files_dir = Path(appdata) / "osu" / "files"
            logs_dir = Path(appdata) / "osu" / "logs"

            # 0. Check Tosu memory reader first (fastest, 0ms, 100% exact beatmap path & content)
            try:
                if tosu_client.is_connected:
                    direct_path = tosu_client.get_active_beatmap_file_path()
                    if direct_path and direct_path.exists():
                        logger.info(f"Exact beatmap loaded directly from Tosu file path: {direct_path}")
                        return self.load_beatmap_file(str(direct_path))

                    bm_content = tosu_client.fetch_active_beatmap_content()
                    if bm_content and "[HitObjects]" in bm_content:
                        m_key = f"{tosu_client.beatmap_id}_{tosu_client.beatmap_title}_{tosu_client.file_name}"
                        logger.info(f"Exact beatmap loaded directly from Tosu memory: {tosu_client.beatmap_title}")
                        return self.load_beatmap_from_string(bm_content, key=m_key)
            except Exception as tosu_err:
                logger.debug(f"Tosu beatmap auto-discovery skipped: {tosu_err}")

            # 1. First check osu!lazer runtime log if present (fastest & 100% accurate for osu!lazer)
            if logs_dir.exists() and files_dir.exists():
                try:
                    runtime_logs = sorted(logs_dir.glob("*.runtime.log"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if runtime_logs:
                        with open(runtime_logs[0], "r", encoding="utf-8", errors="ignore") as f:
                            f.seek(0, os.SEEK_END)
                            size = f.tell()
                            f.seek(max(0, size - 131072))
                            tail = f.read()
                        matches = re.findall(r"Game-wide working beatmap updated to (.*)", tail)
                        if matches:
                            working_map = matches[-1].strip()
                            self._last_lazer_working_map = working_map
                            m_diff = re.search(r"\[(.*?)\]", working_map)
                            if m_diff:
                                diff_name = m_diff.group(1).strip()
                                base_part = working_map.split("[")[0].strip()
                                without_mapper = re.sub(r"\s*\([^)]*\)$", "", base_part).strip()
                                title_hint = without_mapper.split(" - ", 1)[-1].strip() if " - " in without_mapper else without_mapper
                                # Instant In-Memory Index Lookup (<0.001ms)
                                cand = tosu_client.find_file_by_title_diff(title_hint, diff_name)
                                if cand and cand.exists():
                                    logger.info(f"Exact beatmap matched from disk index '{working_map}': {cand}")
                                    return self.load_beatmap_file(str(cand))
                except Exception as e:
                    logger.debug(f"osu!lazer runtime log beatmap discovery skipped: {e}")

            # 2. Try to read active game window title (e.g. "osu! - Artist - Title [Difficulty]" for osu! stable)
            try:
                import ctypes
                desk = ctypes.windll.user32.OpenInputDesktop(0, False, 0x01FF)
                if desk:
                    ctypes.windll.user32.SetThreadDesktop(desk)
                target_title = None
                WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
                def _cb(hwnd, _):
                    nonlocal target_title
                    length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
                        val = buff.value
                        if val.lower().startswith("osu") and "[" in val:
                            target_title = val
                            return False
                    return True
                ctypes.windll.user32.EnumWindows(WNDENUMPROC(_cb), 0)

                if target_title and files_dir.exists():
                    m = re.search(r'osu!\s*-\s*(.*?)\s*\[(.*?)\]', target_title)
                    if m:
                        song_info = m.group(1).strip()
                        diff_name = m.group(2).strip()
                        artist_hint = ""
                        title_hint = song_info.lower()
                        if " - " in song_info:
                            parts = song_info.split(" - ", 1)
                            artist_hint = parts[0].strip().lower()
                            title_hint = parts[1].strip().lower()

                        for root, _, files in os.walk(str(files_dir)):
                            for f in files:
                                fp = os.path.join(root, f)
                                if 1000 < os.path.getsize(fp) < 5_000_000:
                                    try:
                                        with open(fp, 'r', encoding='utf-8', errors='ignore') as fp_in:
                                            head = fp_in.read(2048)
                                            if f'Version:{diff_name}' in head:
                                                # In standard mode, reject mania/taiko/catch maps
                                                if "Mode: 1" in head or "Mode: 2" in head or "Mode: 3" in head:
                                                    continue
                                                head_lower = head.lower()
                                                if title_hint and title_hint not in head_lower:
                                                    continue
                                                if artist_hint and artist_hint not in head_lower:
                                                    continue
                                                logger.info(f"Exact beatmap matched from window title '{target_title}': {fp}")
                                                return self.load_beatmap_file(fp)
                                    except Exception:
                                        pass
            except Exception as e:
                logger.debug(f"Window title beatmap match skipped: {e}")

        except Exception as e:
            logger.warning(f"Auto-discovery of beatmap failed: {e}")
        return False

    def _run_loop(self) -> None:
        """Continuous real-time loop running at ~1000-2000Hz (0.5ms precision)."""
        logger.info("OsuGameplayApp session worker loop started at 2x polling rate (2000Hz).")
        last_error_time = 0.0
        last_error_msg = ""
        while not self._stop_event.is_set():
            try:
                self.step()
                time.sleep(0.0005)
            except Exception as e:
                now_s = time.time()
                err_msg = str(e)
                if (now_s - last_error_time) > 2.0 or err_msg != last_error_msg:
                    last_error_time = now_s
                    last_error_msg = err_msg
                    logger.error(f"Error in OsuGameplayApp step: {e}", exc_info=True)
                time.sleep(0.02)
        logger.info("OsuGameplayApp session worker loop ended.")

    def start(
        self,
        session_id: str = "session_01",
        mode_override: Optional[GameMode] = None,
        run_background_thread: bool = True,
    ) -> bool:
        """Start gameplay session."""
        if self._is_running:
            return True

        is_live_backend = not getattr(getattr(self, "controller", None), "backend", None).__class__.__name__.startswith("Mock") and "PYTEST_CURRENT_TEST" not in os.environ
        if is_live_backend:
            # Bring osu! game window to foreground focus so SendInput targets the game
            try:
                osu_capture.focus_osu_window()
            except Exception as e:
                logger.debug(f"Could not focus osu! window on start: {e}")

            # Ensure Live2D bridge worker is active
            try:
                osu_vts_bridge.start()
            except Exception as e:
                logger.debug(f"Could not start osu_vts_bridge: {e}")

        if mode_override:
            mode_detection_service.set_user_override(mode_override)
        else:
            mode_detection_service.evaluate_mode(beatmap_mode=GameMode.STANDARD, confidence=1.0)

        is_live_backend = not self.controller.backend.__class__.__name__.startswith("Mock")
        if is_live_backend and run_background_thread:
            tosu_client.start(auto_launch=True)

        # Auto-discover beatmap from osu!lazer library if none loaded
        if self._timeline is None:
            self._try_auto_discover_beatmap()

        self._timeline_anchor_ns = clock.now_ns()
        self._stop_event.clear()
        self._is_running = True
        self._session_id = session_id
        self._current_session_start_time = time.time()
        self._session_result_logged = False
        self._reported_fail = False
        self._last_step_miss_count = 0
        self._session_reset("start_session")
        self._last_combo_milestone = 0
        self._last_reported_combo = 0
        self._actions_dispatched_total = 0
        self._last_optical_hit_ms = 0.0
        self._last_optical_don_hand = "left"
        self._last_optical_katsu_hand = "left"
        self._last_hit_by_key = {"z": 0.0, "x": 0.0}
        self._last_alarm_time = 0.0
        self._last_thought_time_s = time.time()
        self._last_tosu_time_ms = -1.0
        self._tosu_anchor_synced = False
        self._timeline_synced_from_optics = False
        self._active_optical_target_id = None
        self._optical_cursor_x = None
        self._optical_cursor_y = None
        self._last_processed_frame_seq = -1
        frame_channel.clear()

        # Update playfield resolution and capture region to match auto-detected game window BEFORE starting capture
        initial_reg = getattr(self.capture, "_region", None)
        if initial_reg is None and not self.controller.backend.__class__.__name__.startswith("Mock"):
            initial_reg = osu_capture.find_osu_window_region()
        if initial_reg:
            self.capture.update_region(initial_reg)
            r_left, r_top, r_right, r_bottom = initial_reg
            w = r_right - r_left
            h = r_bottom - r_top
            if w > 200 and h > 200:
                self.preprocessor.update_resolution(w, h)
                self.preprocessor.update_window_position(r_left, r_top)
                self.detector.update_bounds(self.preprocessor.bounds)
                logger.info(f"OsuGameplayApp calibrated resolution to game window: {w}x{h} at ({r_left}, {r_top})")

        self.telemetry.start_session(session_id)
        if is_live_backend:
            try:
                timing_calibrator.calibrate_for_bpm(160.0, is_memory_mode=True)
            except Exception:
                pass
        if run_background_thread:
            # Zero-latency: Camera capture disabled completely, driven directly by Tosu memory
            if is_live_backend and not tosu_client.is_connected:
                tosu_client.start(auto_launch=True)

        # Probe initial game window title immediately on start for instant menu immunity
        if not self.controller.backend.__class__.__name__.startswith("Mock"):
            try:
                h_osu = osu_capture.find_osu_window_hwnd()
                if h_osu:
                    length = ctypes.windll.user32.GetWindowTextLengthW(h_osu)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(h_osu, buff, length + 1)
                        self._last_seen_game_title = buff.value
            except Exception:
                pass

        # Update system state to PLAYING
        state_store.mutate_domain(
            "system",
            lambda curr: SystemState(
                state=OperatingState.PLAYING,
                cpu_load=curr.cpu_load,
                fps=curr.fps,
                sequence_id=curr.sequence_id + 1,
                updated_at_ns=clock.now_ns(),
            ),
            caller="system",
        )

        # Emit enthusiastic start emotion to Live2D and log thought
        self._emit_emotion_and_thought(
            emotion="excited",
            intensity=1.0,
            duration=4.0,
            text="Dạ Ba Roshi, Shiori bắt đầu vào trận osu rồi nè! Con sẽ tập trung nhịp điệu và phản xạ tốc độ cao 240 FPS! Ba cùng cổ vũ con gái nha! ♡",
            reason="osu_start",
        )

        # Start real-time gameplay background thread if requested
        if run_background_thread:
            self._session_thread = threading.Thread(
                target=self._run_loop,
                name="OsuGameplayAppLoop",
                daemon=True,
            )
            self._session_thread.start()

        logger.info(f"OsuGameplayApp started session '{session_id}' (background={run_background_thread}).")
        return True

    def _pre_aim_first_object(self) -> None:
        """Pre-aim cursor to the first hit object so Shiori is already on target at song start."""
        if not self._timeline or not getattr(self._timeline, "_objects", None) or not getattr(self, "auto_aim_enabled", True):
            return
        first_obj = self._timeline._objects[0]
        if not first_obj:
            return
        sx, sy = self.preprocessor.osu_to_screen(first_obj.x, first_obj.y)
        try:
            self.controller.move_cursor(float(sx), float(sy))
            std_brain = self.planner.get_brain(GameMode.STANDARD)
            if std_brain:
                std_brain._cursor_x = float(first_obj.x)
                std_brain._cursor_y = float(first_obj.y)
            self._pre_aimed_first_note = True
            logger.info(f"Pre-aimed cursor to first hit object at ({sx}, {sy}) [osu: {first_obj.x}, {first_obj.y} at {first_obj.time_ms}ms]")
        except Exception as e:
            logger.debug(f"Pre-aim failed: {e}")

    def _session_reset(self, reason: str = "", reset_metrics: bool = True) -> None:
        """Universal session reset: clears all in-flight state between songs, pauses, or game transitions."""
        if reason:
            logger.info(f"[SessionReset] {reason}")
        self.planner.reset()
        self.queue.clear()
        self.tracker._tracks.clear()
        self.controller.emergency_release_all()
        self._last_key_press_time_ms.clear()
        self._pre_aimed_first_note = False
        if reset_metrics:
            self.feedback.metrics.reset()
            try:
                timing_calibrator.reset_offset()
            except Exception:
                pass
        if hasattr(self.reconciler, "world_manager") and hasattr(self.reconciler.world_manager, "clear"):
            self.reconciler.world_manager.clear()
        if hasattr(self.planner, "_world_manager") and hasattr(self.planner._world_manager, "clear"):
            self.planner._world_manager.clear()
        if hasattr(self, "world_manager") and hasattr(self.world_manager, "clear"):
            self.world_manager.clear()

    def start_session(
        self,
        session_id: str = "session_01",
        beatmap_path: Optional[str] = None,
        mode_override: Optional[GameMode] = None,
        run_background_thread: bool = True,
    ) -> bool:
        """Alias for start() with optional beatmap file loading."""
        if beatmap_path:
            self.load_beatmap_file(beatmap_path)
        return self.start(session_id=session_id, mode_override=mode_override, run_background_thread=run_background_thread)

    def step(self, simulated_time_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute one iteration of the real-time gameplay loop:
        1. Capture frame & detect/track.
        2. Reconcile with Beatmap.
        3. Request safety permission lease.
        4. Plan rolling-horizon actions.
        5. Dispatch due actions from Input Queue through Input Controller.
        6. Emit Live2D emotion reactions & thoughts on milestones.
        """
        now_ns = clock.now_ns()
        now_sec = time.time()
        lazer_is_playing = False

        # Active Game Window Watchdog: ultra-fast HWND validation using IsWindow (<0.001ms), only scan every 1.5s
        is_live_backend = not self.controller.backend.__class__.__name__.startswith("Mock")
        if is_live_backend and simulated_time_ms is None and self._is_running:
            now_s = time.time()
            cached_hwnd = getattr(self, "_cached_osu_hwnd", None)
            if cached_hwnd and ctypes.windll.user32.IsWindow(cached_hwnd):
                osu_hwnd = cached_hwnd
            elif (now_s - getattr(self, "_last_hwnd_scan_time", 0.0)) > 1.5:
                self._last_hwnd_scan_time = now_s
                osu_hwnd = osu_capture.find_osu_window_hwnd()
                self._cached_osu_hwnd = osu_hwnd
            else:
                osu_hwnd = cached_hwnd

            if not osu_hwnd:
                if self.controller._key_down_times or self.controller._mouse_down_times or self.controller._release_heap:
                    self.controller.emergency_release_all()
                if not tosu_client.is_connected:
                    if len(self.queue._heap) > 0 or len(self.planner._committed_actions) > 0:
                        self._session_reset("window-disconnected", reset_metrics=False)
                    if getattr(self, "_was_window_connected", True):
                        self._was_window_connected = False
                        logger.info("Chưa mở osu! -> Chế độ CHỜ an toàn (0% chiếm dụng chuột & bàn phím).")
                    if "PYTEST_CURRENT_TEST" not in os.environ:
                        time.sleep(0.20)
                    return {}
                if getattr(self, "_last_missing_game_time", 0.0) == 0.0:
                    self._last_missing_game_time = now_s
                elif (now_s - self._last_missing_game_time) > 3.0:
                    if len(self.queue._heap) > 0 or len(self.planner._committed_actions) > 0:
                        self._session_reset("window-disconnected", reset_metrics=False)
                    if getattr(self, "_was_window_connected", True):
                        self._was_window_connected = False
                        logger.info("Cửa sổ osu! đang đóng hoặc mất kết nối -> Chuyển sang chế độ chờ tự động kết nối & focus lại khi mở game.")
                    time.sleep(0.20)
                    return {}
            else:
                self._last_missing_game_time = 0.0
                # Tự động kết nối lại Tosu & Focus thẳng vào cửa sổ game osu! ngay khi phát hiện cửa sổ vừa mở/kết nối lại!
                if not getattr(self, "_was_window_connected", True):
                    self._was_window_connected = True
                    if not tosu_client.is_connected:
                        tosu_client.start(auto_launch=True)
                    osu_capture.focus_osu_window()
                    if hasattr(self.controller, "backend"):
                        self.controller.backend._was_osu_active = False
                    self._last_region_sync_time = 0.0
                    if self._timeline is None:
                        self._try_auto_discover_beatmap()
                    logger.info(f"Đã tự động kết nối lại & focus màn hình game osu! (HWND {osu_hwnd}).")

                # Song Title change detection (auto-reset session anchors when restarting or picking new song)
                try:
                    length = ctypes.windll.user32.GetWindowTextLengthW(osu_hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        ctypes.windll.user32.GetWindowTextW(osu_hwnd, buff, length + 1)
                        curr_title = buff.value
                        last_title = getattr(self, "_last_seen_game_title", "")
                        if curr_title != last_title:
                            self._last_seen_game_title = curr_title
                            if last_title != "" and " - " in curr_title and "[" in curr_title:
                                if getattr(self, "_was_active_play", False) or tosu_client.is_playing:
                                    logger.info(f"Song transition/restart detected: '{curr_title}' -> Resetting timeline anchors & discovering beatmap!")
                                    leadin_ns = int(getattr(self, "_current_lazer_leadin_ms", 0.0) * 1_000_000)
                                    self._timeline_anchor_ns = (now_ns - leadin_ns) if leadin_ns < 0 else now_ns
                                    self._timeline_synced_from_optics = bool(leadin_ns < 0)
                                    self._tosu_anchor_synced = False
                                    self._active_optical_target_id = None
                                    self._optical_cursor_x = None
                                    self._optical_cursor_y = None
                                    self._actions_dispatched_total = 0
                                    self._last_tosu_map_key = ""
                                    self._session_reset("song-transition")
                                self._try_auto_discover_beatmap()
                except Exception:
                    pass

                # Periodic window region sync (ensures accurate desktop coordinates even if game window moved)
                now_s = time.time()
                if (now_s - getattr(self, "_last_region_sync_time", 0.0)) > 1.5:
                    self._last_region_sync_time = now_s
                    region = osu_capture.find_osu_window_region()
                    if region:
                        self.capture.update_region(region)
                        ol, ot, oright, obottom = region
                        ow = oright - ol
                        oh = obottom - ot
                        if ow > 200 and oh > 200:
                            self.preprocessor.update_window_position(ol, ot)
                            if ow != self.preprocessor._screen_width or oh != self.preprocessor._screen_height:
                                self.preprocessor.update_resolution(ow, oh)
                                self.detector.update_bounds(self.preprocessor.bounds)

                # osu!lazer runtime state & beatmap change detection
                lazer_is_playing, lazer_working_map = self._check_lazer_gameplay_state(now_s)
                lazer_player_screen = getattr(self, "_current_lazer_player_screen", "")
                lazer_leadin_ms = getattr(self, "_current_lazer_leadin_ms", 0.0)

                is_new_map = bool(lazer_working_map and lazer_working_map != getattr(self, "_last_lazer_working_map", ""))
                is_fresh_screen = bool(lazer_is_playing and lazer_player_screen and lazer_player_screen != getattr(self, "_last_seen_player_screen", ""))
                is_delayed_leadin = bool(
                    lazer_is_playing
                    and lazer_player_screen
                    and lazer_leadin_ms < 0.0
                    and getattr(self, "_last_synced_leadin_ms", 0.0) == 0.0
                )

                if is_new_map or is_fresh_screen or is_delayed_leadin:
                    if is_new_map:
                        self._last_lazer_working_map = lazer_working_map
                        logger.info(f"osu!lazer beatmap change detected: '{lazer_working_map}' -> Discovering beatmap!")
                        self._try_auto_discover_beatmap()
                        self._timeline_synced_from_optics = False
                        self._active_optical_target_id = None
                        self._optical_cursor_x = None
                        self._optical_cursor_y = None
                        self._actions_dispatched_total = 0
                        self._session_reset("beatmap-change")
                    if (is_fresh_screen or is_new_map or is_delayed_leadin) and lazer_is_playing and lazer_player_screen:
                        self._last_seen_player_screen = lazer_player_screen
                        self._last_synced_leadin_ms = lazer_leadin_ms
                        self._lazer_is_paused = False
                        self._lazer_resumed = False
                        self._pause_start_ns = 0
                        logger.info(f"osu!lazer {lazer_player_screen} entered (leadin={lazer_leadin_ms}ms) -> Synchronizing timeline anchor to song start!")
                        # Initial lead-in estimation from lazer log (will be ground-truth locked by Tosu memory as soon as tosu_t > 0)
                        self._timeline_anchor_ns = now_ns - int(lazer_leadin_ms * 1_000_000)
                        self._timeline_synced_from_optics = True
                        self._tosu_anchor_synced = False  # Allow Tosu memory reader to perform exact ground-truth lock on first positive tick
                        self._last_tosu_time_ms = lazer_leadin_ms
                        self._active_optical_target_id = None
                        self._optical_cursor_x = None
                        self._optical_cursor_y = None
                        self._actions_dispatched_total = 0
                        self._session_reset("beatmap-change")

                        # Pre-aim cursor to the first hit object of the map so Shiori is directly on target at song start!
                        self._pre_aim_first_object()
                elif getattr(self, "_lazer_resumed", False):
                    # Resume within the same player screen
                    self._lazer_resumed = False
                    self._tosu_anchor_synced = False
                    pause_duration_ns = now_ns - getattr(self, "_pause_start_ns", now_ns)
                    if pause_duration_ns > 0:
                        self._timeline_anchor_ns += pause_duration_ns
                    self._actions_dispatched_total = 0
                    self._session_reset("gameplay-resume", reset_metrics=False)
                    resumed_time_ms = (now_ns - self._timeline_anchor_ns) / 1_000_000.0
                    logger.info(
                        f"Gameplay RESUMED -> Adjusted timeline anchor by +{pause_duration_ns / 1_000_000.0:.1f}ms "
                        f"(resumed at song time: {resumed_time_ms:.1f}ms)!"
                    )
                elif not lazer_player_screen:
                    # Player completely left gameplay screen (e.g. SongSelect, Results, MainMenu)
                    self._last_seen_player_screen = ""
                    self._timeline_synced_from_optics = False
        else:
            lazer_is_playing = False

        # Authoritative Game Mode Synchronization:
        # Operating strictly in pure osu! Standard mode.
        active_mode = GameMode.STANDARD

        if active_mode and state_store.mode.mode != active_mode and not state_store.mode.user_overridden:
            old_mode = state_store.mode.mode
            mode_detection_service.evaluate_mode(beatmap_mode=active_mode, confidence=1.0)
            if old_mode != GameMode.UNKNOWN:
                self._session_reset("mode-change")

        is_active_play = False
        if simulated_time_ms is not None:
            is_active_play = True
            current_time_ms = simulated_time_ms
            is_new_sim_anchor = (
                not getattr(self, "_sim_anchor_set", False)
                or current_time_ms < getattr(self, "_last_sim_time_ms", 0.0)
                or abs(current_time_ms - getattr(self, "_last_sim_time_ms", 0.0)) > 3000.0
            )
            if is_new_sim_anchor:
                if current_time_ms < getattr(self, "_last_sim_time_ms", 0.0):
                    self._actions_dispatched_total = 0
                    self._session_reset("sim-time-rewind")
                self._timeline_anchor_ns = now_ns - int(simulated_time_ms * 1_000_000)
                self._sim_anchor_set = True
            self._last_sim_time_ms = current_time_ms
        elif tosu_client.is_connected:
            # High-precision live audio clock advancement detection
            raw_tosu_time = tosu_client.time_ms
            last_adv_t = getattr(self, "_last_tosu_advancing_time", -999999.0)
            if tosu_client.is_playing and raw_tosu_time > (last_adv_t + 0.5):
                self._last_tosu_advancing_time = raw_tosu_time
                self._last_tosu_clock_advancing_ns = now_ns
                if getattr(self, "_lazer_is_paused", False):
                    self._lazer_is_paused = False
                    self._lazer_resumed = True
                    self._tosu_anchor_synced = False
                    logger.info("osu!lazer gameplay RESUME confirmed: audio clock advancing!")

            # Authoritative active play detection: tosu_client memory or lazer runtime log, accounting for pause
            is_paused = getattr(self, "_lazer_is_paused", False)
            if tosu_client.is_connected:
                # Tosu memory is 0ms ground truth: state == 2 is strictly Playing.
                # SongSelect (state 5), Menu (state 0), Results (state 7) are NEVER active play!
                is_active_play = tosu_client.is_playing and not is_paused
            else:
                is_active_play = lazer_is_playing and not is_paused

            # State transition detection: playing -> standby (song finished or results screen)
            was_active = getattr(self, "_was_active_play", False)
            self._was_active_play = is_active_play

            # Only report song completion when match genuinely finishes (NOT during pause, NOT during early retry/abort)
            if was_active and not is_active_play and not is_paused:
                if not getattr(self, "_reported_fail", False) and not getattr(self, "_reported_pass", False):
                    has_hits = (self._actions_dispatched_total > 5 or self.feedback.metrics.total_hits > 5)
                    bm_end_ms = getattr(self._timeline, "end_time_ms", 0.0) if self._timeline else 0.0
                    last_t_ms = getattr(self, "_last_tosu_time_ms", 0.0)
                    is_near_end = (bm_end_ms > 0 and last_t_ms >= (bm_end_ms - 3000.0))
                    curr_screen = getattr(self, "_current_lazer_player_screen", "")
                    is_results = "Results" in curr_screen or getattr(tosu_client, "grade", "") in ("SS", "S", "A", "B", "C", "D")
                    if not self._timeline:
                        is_near_end = is_results or last_t_ms > 15000.0

                    if has_hits and (is_near_end or is_results):
                        self._reported_pass = True
                        self._handle_song_completed()

            if is_paused:
                # Paused in player screen: sleep briefly, keep future planned actions and state intact!
                time.sleep(0.005)
                return {}

            if not is_active_play:
                # Standby: Player is in song select, results, or menu -> 0 inputs!
                if tosu_client.time_ms is not None and not math.isnan(tosu_client.time_ms):
                    self._last_standby_tosu_time_ms = float(tosu_client.time_ms)
                self._tosu_anchor_synced = False
                self._timeline_synced_from_optics = False
                self._last_tosu_time_ms = -1.0
                if len(self.queue._heap) > 0 or len(self.planner._committed_actions) > 0:
                    self._session_reset("standby-flush", reset_metrics=False)
                # Sleep briefly to free CPU cycles for Live2D and Discord during song selection
                time.sleep(0.005)
                return {}
            else:
                if not was_active:
                    self._reported_fail = False
                    self._reported_pass = False
                    self._last_step_miss_count = 0
                    if is_live_backend and simulated_time_ms is None:
                        osu_capture.focus_osu_window()
                        if hasattr(self.controller, "backend"):
                            self.controller.backend._was_osu_active = False
                elif is_live_backend and simulated_time_ms is None:
                    # Đang trong màn chơi thực tế: nếu cửa sổ osu! bị mất focus ngoài ý muốn (popup/notification), tự động focus lại
                    now_f = time.time()
                    human_until = getattr(getattr(self.controller, "backend", None), "_human_override_until_s", 0.0)
                    if now_f >= human_until and (now_f - getattr(self, "_last_gameplay_refocus_s", 0.0)) > 0.45:
                        cur_hwnd = getattr(self, "_cached_osu_hwnd", None)
                        if cur_hwnd and ctypes.windll.user32.GetForegroundWindow() != cur_hwnd:
                            self._last_gameplay_refocus_s = now_f
                            osu_capture.focus_osu_window()

            # Sync active beatmap directly from Tosu memory server (supports both ranked and unranked ID=0)
            valid_tosu_bm = bool(
                tosu_client.beatmap_id > 0
                or (tosu_client.file_name and tosu_client.file_name.strip())
                or (tosu_client.beatmap_title and tosu_client.beatmap_title.strip() not in ("", " -  []"))
            )
            current_map_key = f"{tosu_client.beatmap_id}_{tosu_client.beatmap_title}_{tosu_client.file_name}" if valid_tosu_bm else ""
            now_sec = time.time()
            last_map_attempt = getattr(self, "_last_map_attempt_s", 0.0)
            needs_map_sync = (
                valid_tosu_bm
                and (
                    current_map_key != getattr(self, "_last_tosu_map_key", "")
                    or (self._timeline is None and (now_sec - last_map_attempt > 0.5))
                )
            )
            if needs_map_sync:
                self._last_map_attempt_s = now_sec
                # Reset user override khi chọn bài hát mới để chế độ chơi tự động theo bài hát
                mode_detection_service.clear_user_override()

                loaded = False
                # Ưu tiên nạp trực tiếp từ file path trên đĩa (<0.3ms)
                direct_path = tosu_client.get_active_beatmap_file_path()
                if direct_path and direct_path.exists():
                    loaded = self.load_beatmap_file(str(direct_path))
                    if loaded:
                        self._last_tosu_map_key = current_map_key
                        self._last_tosu_beatmap_id = tosu_client.beatmap_id
                        logger.info(f"Loaded active beatmap directly from disk: {direct_path.name}")

                if not loaded:
                    bm_content = tosu_client.fetch_active_beatmap_content()
                    if bm_content:
                        loaded = self.load_beatmap_from_string(bm_content, key=current_map_key)
                        if loaded:
                            self._last_tosu_map_key = current_map_key
                            self._last_tosu_beatmap_id = tosu_client.beatmap_id
                            logger.info(f"Loaded active beatmap directly from Tosu memory: ID={tosu_client.beatmap_id} ({tosu_client.beatmap_title})")

                if not loaded and self._timeline is None:
                    # Fallback auto-discovery from local disk or lazer logs
                    if self._try_auto_discover_beatmap():
                        self._last_tosu_map_key = current_map_key

            # High-precision Ground-Truth game time directly from Tosu memory (handles negative intro lead-in and zero ms)
            if is_active_play and tosu_client.is_connected and tosu_client.is_playing and tosu_client.time_ms is not None and not math.isnan(tosu_client.time_ms):
                tosu_precise_t = tosu_client.get_interpolated_time_ms(now_ns)
                last_tosu_t = getattr(self, "_last_tosu_time_ms", 0.0)
                is_rewind = (last_tosu_t > 0.0) and (tosu_precise_t < (last_tosu_t - 250.0))

                first_obj_ms = (
                    float(self._timeline._objects[0].time_ms)
                    if (self._timeline and getattr(self._timeline, "_objects", None))
                    else 0.0
                )
                total_tosu_hits = (
                    getattr(tosu_client, "hits_300", 0)
                    + getattr(tosu_client, "hits_100", 0)
                    + getattr(tosu_client, "hits_50", 0)
                    + getattr(tosu_client, "hits_miss", 0)
                )
                standby_t = getattr(self, "_last_standby_tosu_time_ms", -1.0)
                is_continuing_preview = (
                    standby_t > 500.0
                    and tosu_precise_t >= (standby_t - 200.0)
                    and (tosu_precise_t - standby_t) < 10000.0
                )
                is_unjudged_far_future = (
                    total_tosu_hits == 0
                    and getattr(tosu_client, "current_combo", 0) == 0
                    and tosu_precise_t > max(3500.0, first_obj_ms + 1500.0)
                )
                # ponytail: preview-audio guard assumes song entry starts before first_obj_ms+1500ms; upgrade if mid-song practice seek with 0 hits is added.
                is_preview_audio = (
                    not self._tosu_anchor_synced
                    and tosu_precise_t > 500.0
                    and (is_continuing_preview or is_unjudged_far_future)
                )

                # Intro countdown protection:
                # If we have an active lead-in countdown (expected_t < 20ms or leadin < 0) and Tosu reports <= 0.0ms,
                # the audio clock in osu! has not reached 0ms yet. Do NOT overwrite anchor with 0.0 or treat as drift!
                has_leadin = getattr(self, "_timeline_synced_from_optics", False) or (self._timeline_anchor_ns > now_ns)
                expected_t = (now_ns - self._timeline_anchor_ns) / 1_000_000.0 if (self._tosu_anchor_synced or has_leadin) else tosu_precise_t
                is_intro_countdown = (expected_t < 20.0) and (tosu_precise_t <= 0.0) and (has_leadin or self._timeline_anchor_ns > now_ns)

                # Protect timeline anchor when Tosu transiently reports 0.0ms while game is already deep in progress
                is_suspect_zero = (
                    tosu_client.time_ms == 0.0
                    and self._tosu_anchor_synced
                    and ((now_ns - self._timeline_anchor_ns) / 1_000_000.0) > 1000.0
                    and not is_rewind
                    and not (last_tosu_t > 1000.0)
                )

                if not is_suspect_zero and not is_intro_countdown and not is_preview_audio:
                    # Check if audio clock is frozen during active play (e.g. pause in stable or unpause countdown)
                    # NOTE: Do NOT treat negative intro countdown (<= 0.0) as paused!
                    if tosu_precise_t > 0.0 and abs(tosu_precise_t - last_tosu_t) < 0.001 and self._tosu_anchor_synced:
                        now_s = time.time()
                        if (now_s - getattr(self, "_last_tosu_tick_time", now_s)) > 0.25:
                            # Audio clock is paused! Keep anchor pinned to tosu_precise_t and release keys
                            self.controller.emergency_release_all()
                            self._timeline_anchor_ns = now_ns - int(tosu_precise_t * 1_000_000)
                    else:
                        self._last_tosu_tick_time = time.time()

                    drift_ms = tosu_precise_t - expected_t

                    # Zero-drift synchronizer with deadband & gentle PLL damping to eliminate HTTP jitter
                    if not self._tosu_anchor_synced:
                        if tosu_precise_t > 0.0:
                            logger.info(
                                f"First positive audio clock tick detected from Tosu ({tosu_precise_t:.1f}ms) "
                                "-> Locking ground-truth timeline anchor & synchronizing opening hit planner!"
                            )
                            self._timeline_anchor_ns = now_ns - int(tosu_precise_t * 1_000_000)
                            self._tosu_anchor_synced = True
                            self._last_tosu_time_ms = tosu_precise_t
                            # Cleanly reset planner and input queue so opening hit objects
                            # (like firstObj at 161ms) are freshly planned with zero drift against real audio clock
                            self.planner.reset()
                            self.queue.clear()
                            if hasattr(self.reconciler, "world_manager") and hasattr(self.reconciler.world_manager, "clear"):
                                self.reconciler.world_manager.clear()
                            if self._timeline and getattr(self._timeline, "_objects", None):
                                first_obj = self._timeline._objects[0]
                                if first_obj:
                                    std_brain = self.planner.get_brain(GameMode.STANDARD)
                                    if std_brain:
                                        std_brain._cursor_x = float(first_obj.x)
                                        std_brain._cursor_y = float(first_obj.y)
                        elif tosu_precise_t < 0.0 and not has_leadin and self._timeline_anchor_ns <= now_ns:
                            self._timeline_anchor_ns = now_ns - int(tosu_precise_t * 1_000_000)
                            self._tosu_anchor_synced = True
                            self._last_tosu_time_ms = tosu_precise_t
                    elif is_rewind or abs(drift_ms) > 40.0:
                        if is_rewind or abs(drift_ms) > 150.0:
                            reason = "Song rewind/retry" if is_rewind else f"Timeline seek/skip (drift={drift_ms:+.1f}ms)"
                            logger.info(f"{reason} -> Resyncing planner and flushing obsolete actions!")
                            self._actions_dispatched_total = 0
                            self._session_reset(reason, reset_metrics=is_rewind)
                            self._pre_aim_first_object()
                            if is_rewind:
                                self._tosu_anchor_synced = False

                        leadin_t = getattr(self, "_current_lazer_leadin_ms", 0.0)
                        if is_rewind and leadin_t < 0.0 and tosu_precise_t <= 0.0:
                            self._timeline_anchor_ns = now_ns - int(leadin_t * 1_000_000)
                        else:
                            self._timeline_anchor_ns = now_ns - int(tosu_precise_t * 1_000_000)
                        self._last_tosu_time_ms = tosu_precise_t
                    elif abs(drift_ms) > 3.0:
                        # Gentle PLL low-pass filter (2% per tick) smoothly corrects crystal oscillator drift without jittering the timeline
                        self._timeline_anchor_ns -= int(drift_ms * 0.02 * 1_000_000)
                        self._last_tosu_time_ms = tosu_precise_t
                    else:
                        self._last_tosu_time_ms = tosu_precise_t

            # Monotonic sub-millisecond game time
            current_time_ms = (now_ns - self._timeline_anchor_ns) / 1_000_000.0
            if is_active_play and tosu_client.is_connected and not self._tosu_anchor_synced:
                if tosu_client.time_ms is not None and (tosu_client.time_ms == 0.0 or locals().get("is_preview_audio", False)):
                    # Hold gameplay clock in pre-start lead-in (<= -250ms, outside 150ms commit window)
                    # while Tosu is still on preview audio or clamped at 0.0ms during negative intro countdown.
                    current_time_ms = min(current_time_ms, -250.0)

            # Sync real in-game combo, accuracy, max combo, and hit distributions directly from Tosu memory
            self.feedback.metrics.current_combo = tosu_client.current_combo
            self.feedback.metrics.accuracy = tosu_client.accuracy
            self.feedback.metrics.max_combo = max(
                self.feedback.metrics.max_combo,
                getattr(tosu_client, "max_combo", 0),
                tosu_client.current_combo,
            )
            self.feedback.metrics.count_300 = getattr(tosu_client, "hits_300", 0)
            self.feedback.metrics.count_100 = getattr(tosu_client, "hits_100", 0)
            self.feedback.metrics.count_50 = getattr(tosu_client, "hits_50", 0)
            self.feedback.metrics.count_miss = getattr(tosu_client, "hits_miss", 0)

            # Closed-Loop Real-Time Adaptive Timing Calibration from Tosu memory hit errors
            if is_active_play and getattr(tosu_client, "hit_errors", None):
                timing_calibrator.consume_hit_errors(tosu_client.hit_errors)
                self.feedback.record_tosu_hit_errors(tosu_client.hit_errors)
        else:
            if is_live_backend and simulated_time_ms is None and not lazer_is_playing:
                # Tosu chưa kết nối và osu!lazer không ở trong màn chơi: KHÓA CỨNG 100% KHÔNG ĐƯỢC GÕ PHÍM HAY DI CHUỘT!
                if self.controller._key_down_times or self.controller._mouse_down_times or self.controller._release_heap:
                    self.controller.emergency_release_all()
                if len(self.queue._heap) > 0 or len(self.planner._committed_actions) > 0:
                    self._session_reset("standby-no-tosu", reset_metrics=False)
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    time.sleep(0.05)
                return {}
            current_time_ms = (now_ns - self._timeline_anchor_ns) / 1_000_000.0

        # Hybrid Vision Integration: Asynchronous, non-blocking frame consumption if capture is active
        tracked_objects = []
        if hasattr(self, "capture") and self.capture.is_running:
            if hasattr(self.capture, "frame_channel"):
                frame = self.capture.frame_channel.get_latest()
                if frame is not None:
                    detections = self.detector.detect(frame.image, frame.timestamp_ns)
                    tracked_objects = self.tracker.update(detections, frame.timestamp_ns)
                else:
                    tracked_objects = self.tracker.active_tracks
        elif hasattr(self, "tracker"):
            tracked_objects = self.tracker.active_tracks
        actions_dispatched = 0

        # Mode auto-sync from loaded timeline when mode is still UNKNOWN
        if state_store.mode.mode == GameMode.UNKNOWN and self._timeline is not None:
            first_objs = self._timeline.get_objects_in_window(0.0, 100000.0)
            if first_objs:
                mode_detection_service.evaluate_mode(beatmap_mode=GameMode.STANDARD, confidence=1.0)

        if is_active_play and not getattr(self, "_pre_aimed_first_note", False) and self._actions_dispatched_total == 0:
            self._pre_aim_first_object()

        # 2. Reconciliation
        reconciled = self.reconciler.reconcile(
            current_time_ms=current_time_ms,
            tracked_objects=tracked_objects,
            preprocessor=self.preprocessor,
            timeline=self._timeline,
        )

        # 3. Safety Check
        perm = self.safety.request_action_permission(duration_ms=50)

        if perm.is_allowed:
            # 4. Planning (Beatmap + Timeline Fusion)
            # If beatmap timeline is available, StandardBrain & Planner drive the gameplay:
            # - Minimum-jerk smooth aiming
            # - Key alternation (Z/X)
            # - Slider dragging along curve waypoints with mouse/key held down
            # - Spinner rotation
            if self._timeline is not None:
                self.planner.plan_cycle(
                    current_time_ms=current_time_ms,
                    timeline_anchor_ns=self._timeline_anchor_ns,
                    mode=state_store.mode.mode,
                    observed_cursor_pos=self.controller.cursor_pos,
                )
            else:
                # Active play without loaded timeline: trigger emergency discovery immediately
                if is_active_play and (now_sec - getattr(self, "_last_emergency_disc_s", 0.0) > 0.5):
                    self._last_emergency_disc_s = now_sec
                    logger.warning("Active gameplay detected but timeline is missing! Triggering emergency beatmap discovery...")
                    self._try_auto_discover_beatmap()

                self.planner.plan_cycle(
                    current_time_ms=current_time_ms,
                    timeline_anchor_ns=self._timeline_anchor_ns,
                    mode=GameMode.STANDARD,
                    observed_cursor_pos=self.controller.cursor_pos,
                )

            # 5. Input Dispatch & Machine Learning Adaptive Calibration
            effective_now_ns = (
                (self._timeline_anchor_ns + int(simulated_time_ms * 1_000_000))
                if simulated_time_ms is not None
                else now_ns
            )
            dispatched_keys_in_tick = set()
            while True:
                due_action = self.queue.pop_due(effective_now_ns)
                if due_action is None:
                    break
                # Negative audio time guard: never dispatch key taps/downs into the void during intro lead-in countdown
                if current_time_ms < -20.0 and due_action.action_type in (ActionType.KEY_TAP, ActionType.KEY_DOWN):
                    continue
                # If auto-aim is disabled, skip cursor movement so user has full manual aim control
                if not getattr(self, "auto_aim_enabled", True) and due_action.action_type == ActionType.MOUSE_MOVE:
                    continue

                # Key Safety: In Standard mode, strictly ensure only valid standard keys (z/x) can be dispatched
                k = getattr(due_action, "key", None)
                if k and due_action.action_type in (ActionType.KEY_TAP, ActionType.KEY_DOWN, ActionType.KEY_UP):
                    valid_std_keys = getattr(self.planner.get_brain(GameMode.STANDARD), "keys", ["z", "x"])
                    if state_store.mode.mode == GameMode.STANDARD and k not in valid_std_keys:
                        logger.warning(f"Blocked non-standard key dispatch '{k}' in Standard mode. Allowed keys: {valid_std_keys}")
                        continue

                # Dispatched key tracking: record last press timestamp
                if k and due_action.action_type in (ActionType.KEY_TAP, ActionType.KEY_DOWN):
                    self._last_key_press_time_ms[k] = current_time_ms

                success = self.controller.execute_action(due_action)
                if due_action.status == InputStateStatus.CANCELLED:
                    self.telemetry.log_event(
                        "action_cancelled",
                        {
                            "action_id": due_action.action_id,
                            "target_id": due_action.target_id,
                            "action_type": due_action.action_type.value,
                            "cursor_owner": due_action.cursor_owner.value,
                        },
                    )
                    continue
                if success:
                    actions_dispatched += 1
                    key_pressed = getattr(due_action, "key", "") or getattr(due_action, "button", "") or "aim"
                    self._last_action_desc = f"{key_pressed.upper()} ({due_action.action_id})"
                    self.telemetry.log_event("action_executed", {"action_id": due_action.action_id, "key": key_pressed})
                    if due_action.cursor_owner == CursorOwnerState.SLIDER_FOLLOW:
                        self.telemetry.log_event(
                            "slider_diagnostic",
                            {
                                "action_id": due_action.action_id,
                                "target_id": due_action.target_id,
                                "progress": due_action.metadata.get("progress"),
                                "repeat_index": due_action.metadata.get("repeat_index"),
                                "expected_pos": (due_action.x, due_action.y),
                                "actual_pos": self.controller.cursor_pos,
                                "scheduled_ns": due_action.execute_at_ns,
                                "actual_ns": effective_now_ns,
                                "cursor_owner": due_action.cursor_owner.value,
                                "queue_size": len(self.queue),
                            },
                        )

                    # Hit Feedback Evaluation (evaluated internally when Tosu is not connected, e.g. tests)
                    obj_id = due_action.action_id.replace("act_", "")
                    delta_ms = 0.0
                    hit_result = "300"
                    if not tosu_client.is_connected and due_action.action_type in (ActionType.KEY_TAP, ActionType.KEY_DOWN, ActionType.MOUSE_CLICK):
                        # Timing error is measured against the INTENDED hit (gameplay) time, NOT the
                        # dispatch time. execute_at_ns is lead-compensated (target - lead), so measuring
                        # against it would always yield ~0 and fabricate a perfect "300" for every note.
                        # Without real Tosu/vision feedback this is the planned execution-to-hit error.
                        target_time_ms = float(getattr(due_action, "target_time_ms", 0.0) or 0.0)
                        if target_time_ms > 0.0:
                            target_time_ns = self._timeline_anchor_ns + int(target_time_ms * 1_000_000)
                        else:
                            target_time_ns = getattr(due_action, "execute_at_ns", now_ns)
                        delta_ms = (now_ns - target_time_ns) / 1_000_000.0
                        hit_result = self.feedback.evaluate_hit(
                            obj_id=obj_id,
                            delta_ms=delta_ms,
                            od=state_store.beatmap.od or 5.0,
                            mode=state_store.mode.mode,
                            confidence=due_action.confidence,
                        )

                    self.telemetry.log_event(
                        "hit_evaluated",
                        {
                            "obj_id": obj_id,
                            "key": key_pressed,
                            "result": hit_result,
                            "delta_ms": delta_ms,
                            "adaptive_offset_ms": getattr(self.feedback, "calibrator", None).adaptive_offset_ms if hasattr(self.feedback, "calibrator") else 0.0,
                        },
                    )

        # 6. Zero-Latency Memory Architecture (visual camera recording eliminated)
        # 7. Live2D Reaction & Thought Emission
        self._actions_dispatched_total += actions_dispatched
        current_combo = self.feedback.metrics.current_combo
        current_acc = self.feedback.metrics.accuracy
        now_s = time.time()

        # Update Live2D dynamic tracking (head, eyes, body rhythm)
        if getattr(self, "enable_vtuber_bridge", True):
            try:
                cur_x, cur_y = getattr(self.controller, "_cursor_pos", (800.0, 450.0))
                bounds = getattr(self.preprocessor, "bounds", None)
                win_w = getattr(bounds, "width", 1600.0) if bounds else 1600.0
                win_h = getattr(bounds, "height", 900.0) if bounds else 900.0
                act_bpm = state_store.beatmap.bpm or 180.0
                osu_vts_bridge.update_gameplay_tracking(
                    cursor_x=cur_x,
                    cursor_y=cur_y,
                    window_w=win_w,
                    window_h=win_h,
                    bpm=act_bpm,
                    combo=current_combo,
                )
            except Exception:
                pass

        # Fail / Defeat / Chảo úp đầu ("thừa hiện cái chảo lên đầu và có log tất cả các màn thừa tắt đều log")
        current_misses = getattr(tosu_client, "hits_miss", 0) if tosu_client.is_connected else self.feedback.metrics.count_miss
        last_step_misses = getattr(self, "_last_step_miss_count", 0)
        miss_burst = (current_misses - last_step_misses >= 3) and current_combo == 0 and current_misses >= 3
        self._last_step_miss_count = current_misses

        tosu_failed = getattr(tosu_client, "is_failed", False) or (
            tosu_client.is_connected
            and tosu_client.hp <= 0.0
            and tosu_client.is_playing
            and current_misses >= 3
        )
        if (tosu_failed or miss_burst) and not getattr(self, "_reported_fail", False) and (current_time_ms > 3500.0 and (self._actions_dispatched_total >= 5 or current_misses >= 3)):
            self._reported_fail = True
            self._emit_emotion_and_thought(
                emotion="slapstick",
                intensity=1.0,
                duration=4.5,
                text="Ui da! Bị chảo gõ boong một phát rồi Ba ơi! Nốt dồn nhanh quá con đỡ không kịp... Đau đầu quá à hu hu! 🍳💥",
                reason="fail_slapstick",
            )
            self._log_session_to_memory(result="fail", failure_reason="hp_depleted" if tosu_failed else "miss_streak")
            try:
                b_meta = getattr(self._active_beatmap, "metadata", None)
                osu_discord_reporter.send_game_summary(
                    title=state_store.beatmap.title or (getattr(b_meta, "title", "osu! Beatmap") if b_meta else "osu! Beatmap"),
                    artist=state_store.beatmap.artist or (getattr(b_meta, "artist", "Various Artists") if b_meta else "Various Artists"),
                    difficulty=getattr(b_meta, "version", "Normal") if b_meta else "Normal",
                    mode=state_store.mode.mode.value,
                    accuracy=current_acc,
                    combo=current_combo,
                    max_combo=self.feedback.metrics.max_combo,
                    hits=dict(self.feedback.metrics.hits),
                    commentary="Ui da Ba Roshi ơi! Nốt dồn nhanh quá làm Shiori bị chảo đập trúng đầu rồi! Con sẽ luyện tập lại ngay nè! 🍳",
                    is_failed=True,
                    status="failed",
                    lead_time_ms=getattr(timing_calibrator, "system_latency_ms", 12.0),
                    emotion="slapstick",
                )
            except Exception:
                pass

        # Combo milestones: celebrate on real in-game combo (10, 20, 25, 50, 100, 150, 200, 300, 500)
        milestones = [10, 20, 25, 50, 100, 150, 200, 300, 500]
        reached = [m for m in milestones if current_combo >= m and self._last_combo_milestone < m]
        if reached:
            m = max(reached)
            self._last_combo_milestone = m
            self._last_reported_combo = current_combo
            self._last_thought_time_s = now_s
            if m == 10:
                thought = "Yeah, con bắt nhịp được rồi Ba ơi! 10 combo mở màn suôn sẻ! 🌸"
            elif m == 20:
                thought = "Yeah! 20 combo mượt mà rồi Ba ơi! Con gái bắt đầu vào guồng rồi nè! 🌸"
            elif m == 50:
                thought = "50x COMBO! Con gái của Ba đang gõ cực kỳ chuẩn xác luôn nè! ♡"
            elif m == 100:
                thought = "Oaaa 100x COMBO rồi Ba Roshi ơi! Con đang phiêu theo từng nốt nhạc, tay gõ thoăn thoắt luôn! 🚀🎉"
            elif m >= 200:
                thought = f"{m}x COMBO ĐỈNH CAO! Cảm giác nhịp điệu tuyệt vời quá Ba ơi! Giữ vững chuỗi nào! 💖"
            else:
                thought = f"Combo {m}x rồi nè Ba! Con đang duy trì độ chính xác rất tốt! ♡"

            self._emit_emotion_and_thought(
                emotion="happy" if m < 100 else "excited",
                intensity=1.0,
                duration=3.0,
                text=thought,
                reason="combo_milestone",
            )
            # Discord combo milestone embeds suppressed per user request (only final summary on finish)

        # Combo break detection
        if self._last_reported_combo >= 15 and current_combo == 0:
            self._emit_emotion_and_thought(
                emotion="surprised",
                intensity=0.9,
                duration=2.5,
                text="Ái chà, nốt vừa rồi nhanh quá làm con lỡ mất một nhịp! Con hít thở sâu và lấy lại combo ngay cho Ba xem nè!",
                reason="combo_break",
            )
            self._last_reported_combo = 0
            self._last_combo_milestone = 0
            self._last_thought_time_s = now_s
        elif current_combo > 0:
            self._last_reported_combo = current_combo

        # Periodic rhythm thought (every 12 seconds if playing actively)
        if now_s - self._last_thought_time_s > 12.0:
            self._last_thought_time_s = now_s
            if tosu_client.is_connected and current_combo > 0:
                if current_acc >= 90.0:
                    thought = f"Nhịp điệu đang rất đều, độ chính xác {current_acc:.1f}%! Shiori đang tập trung tối đa đây Ba! ♡"
                    emotion = "happy"
                else:
                    thought = "Nốt dồn nhanh ghê, con đang tập trung đồng bộ từng mili-giây để bắt kịp nhịp! Cố lên Shiori!"
                    emotion = "thinking"
            elif self._actions_dispatched_total > 0:
                thought = f"Con đang đồng bộ nhịp từ bộ nhớ game cực chuẩn (0ms trễ, đã bấm {self._actions_dispatched_total} nốt)! Ba cùng cổ vũ con nha! ♡"
                emotion = "excited"
            else:
                thought = "Shiori đã sẵn sàng đồng bộ trực tiếp từ bộ nhớ game! Ba mở bài hát lên nha! ♡"
                emotion = "thinking"

            self._emit_emotion_and_thought(
                emotion=emotion,
                intensity=0.8,
                duration=2.5,
                text=thought,
                reason="rhythm_flow",
            )

        return {
            "current_time_ms": current_time_ms,
            "reconciled_objects": len(reconciled),
            "permission_allowed": perm.is_allowed,
            "actions_dispatched": actions_dispatched,
            "accuracy": self.feedback.metrics.accuracy,
            "combo": self.feedback.metrics.current_combo,
            "current_emotion": self._current_emotion,
            "last_thought": self._last_thought,
        }

    def stop(self) -> None:
        """Gracefully stop gameplay session and release hardware."""
        self._is_running = False
        self._stop_event.set()
        if self._session_thread and self._session_thread.is_alive():
            if threading.current_thread() != self._session_thread:
                self._session_thread.join(timeout=1.0)
        self._session_thread = None

        self.capture.stop()
        tosu_client.stop()
        mode_detection_service.clear_user_override()
        self.queue.clear()
        self.tracker._tracks.clear()
        self.controller.emergency_release_all()
        self._last_key_press_time_ms.clear()
        self.telemetry.stop_session()
        frame_channel.clear()
        self._last_processed_frame_seq = -1
        try:
            osu_vts_bridge.reset_all()
        except Exception:
            pass

        # Update system state to READY
        state_store.mutate_domain(
            "system",
            lambda curr: SystemState(
                state=OperatingState.READY,
                cpu_load=curr.cpu_load,
                fps=curr.fps,
                sequence_id=curr.sequence_id + 1,
                updated_at_ns=clock.now_ns(),
            ),
            caller="system",
        )

        acc = self.feedback.metrics.accuracy
        max_combo = self.feedback.metrics.max_combo
        total_hits = self.feedback.metrics.total_hits

        # Prioritize Tosu live memory if available
        if tosu_client.is_connected:
            tosu_hits = tosu_client.hits_300 + tosu_client.hits_100 + tosu_client.hits_50 + tosu_client.hits_miss
            if tosu_hits > total_hits:
                total_hits = tosu_hits
                acc = tosu_client.accuracy
                max_combo = tosu_client.max_combo

        # Determine pass vs fail vs quit for persistent Game Memory
        if getattr(self, "_reported_fail", False) or (tosu_client.is_connected and tosu_client.is_failed):
            final_result = "fail"
            final_reason = "hp_zero_or_miss_streak"
            summary_commentary = f"Ui da~ Màn này tiếc quá Ba ơi, nốt dồn nhanh làm con bị chảo úp đầu luôn hu hu! Để con phân tích lại dữ liệu trong bộ nhớ để lần sau gõ chuẩn hơn nha Ba! 🍳💦"
            stop_emotion = "slapstick"
        elif total_hits > 5 and acc >= 60.0:
            final_result = "pass"
            final_reason = "song_completed"
            summary_commentary = f"Phù~ Phiên chơi vừa xong rồi nè Ba ơi! Độ chính xác đạt {acc:.1f}%, combo cao nhất {max_combo}x! Ba thấy con gái biểu diễn có tiến bộ không ạ? ♡"
            stop_emotion = "happy"
        else:
            final_result = "quit"
            final_reason = "manual_quit_or_interrupted"
            summary_commentary = f"Shiori đã dừng phiên chơi an toàn rồi nè Ba ơi! Con đã lưu lại toàn bộ thông tin nhịp gõ vào bộ nhớ để lần sau tối ưu hơn nữa nha! ♡"
            stop_emotion = "neutral"

        had_actual_gameplay = bool(total_hits > 0 or self._actions_dispatched_total > 0 or getattr(self, "_reported_fail", False))
        if not had_actual_gameplay:
            logger.info("OsuGameplayApp stopped from standby (0 hits) -> skipping empty memory/Discord report.")
            return

        self._log_session_to_memory(result=final_result, failure_reason=final_reason)

        self._emit_emotion_and_thought(
            emotion=stop_emotion,
            intensity=1.0,
            duration=4.0,
            text=summary_commentary,
            reason="osu_stop",
        )

        # Dispatch match results to Discord channel 1563921667556773969
        try:
            b_meta = getattr(self._active_beatmap, "metadata", None)
            title = state_store.beatmap.title or (getattr(b_meta, "title", "osu! Session") if b_meta else "osu! Session")
            artist = state_store.beatmap.artist or (getattr(b_meta, "artist", "Various Artists") if b_meta else "Various Artists")
            version = getattr(b_meta, "version", "Normal") if b_meta else "Normal"
            osu_discord_reporter.send_game_summary(
                title=title,
                artist=artist,
                difficulty=version,
                mode=state_store.mode.mode.value,
                accuracy=acc,
                combo=self.feedback.metrics.current_combo,
                max_combo=max_combo,
                hits=dict(self.feedback.metrics.hits),
                commentary=summary_commentary,
                is_failed=(final_result == "fail"),
                status=final_result,
                lead_time_ms=getattr(timing_calibrator, "system_latency_ms", 12.0),
                emotion=stop_emotion,
            )
        except Exception as e:
            logger.debug(f"Failed to report match to Discord: {e}")

        logger.info("OsuGameplayApp stopped successfully.")

    def get_status(self) -> Dict[str, Any]:
        """Summary status for Shiori companion command integration."""
        gpu_stat = gpu_engine.get_hardware_status()
        return {
            "is_running": self._is_running,
            "operating_state": state_store.system.state.value,
            "mode": state_store.mode.mode.value,
            "beatmap_loaded": state_store.beatmap.is_loaded,
            "beatmap_title": state_store.beatmap.title,
            "accuracy": round(self.feedback.metrics.accuracy, 2),
            "combo": self.feedback.metrics.current_combo,
            "max_combo": self.feedback.metrics.max_combo,
            "unstable_rate": round(self.feedback.metrics.unstable_rate, 2),
            "adaptive_offset_ms": round(state_store.timing.adaptive_offset_ms, 2),
            "tosu_connected": tosu_client.is_connected,
            "gpu_accelerated": gpu_stat.get("gpu_accelerated", False),
            "gpu_device": gpu_stat.get("device_name", "NVIDIA GeForce RTX 2060"),
            "gpu_vram_free_mb": gpu_stat.get("free_vram_mb", 0.0),
            "auto_aim_enabled": getattr(self, "auto_aim_enabled", False),
            "fps": round(self.capture.actual_fps, 1),
            "recent_action": self._last_action_desc,
            "current_emotion": self._current_emotion,
            "last_thought": self._last_thought,
        }


osu_app = OsuGameplayApp()
