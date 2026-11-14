"""
Unit Tests for Shiori osu! Game Memory, Self-Tuning, and VTuber Expressions
===========================================================================
Verifies:
1. OsuGameMemory logging, filtering, and performance analysis.
2. Self-tuning lead time adjustments based on historical error patterns.
3. Fast/Hard beatmap detection triggering 'thinking' expression.
4. Combo milestones (20, 50, 100) triggering 'happy' / 'excited'.
5. Thua / Defeat / HP=0 triggering 'slapstick' (chảo úp đầu) emotion.
6. Game session termination logging all outcomes (pass, fail, quit) to memory.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from src.gameplay.osu.app import OsuGameplayApp
from src.gameplay.osu.beatmap.models import BeatmapData, DifficultySection, MetadataSection
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.evaluation.game_memory import OsuGameMemory
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.integrations.tosu_client import tosu_client
from src.gameplay.osu.integrations.vts_bridge import osu_vts_bridge
from src.gameplay.osu.timing.calibration import timing_calibrator


def test_game_memory_logging_and_analysis():
    """Verify OsuGameMemory records sessions and analyzes timing errors accurately."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem_file = Path(tmp_dir) / "test_memory.jsonl"
        memory = OsuGameMemory(memory_file_path=str(mem_file))

        # Log 3 sessions: 1 pass, 1 fail, 1 quit
        memory.log_session({
            "session_id": "sess_1",
            "beatmap_id": 12345,
            "beatmap_title": "Test Song [Insane]",
            "mode": "taiko",
            "result": "pass",
            "accuracy": 98.5,
            "combo": 250,
            "max_combo": 250,
            "miss_count": 0,
            "avg_timing_error_ms": 6.0,  # Hitting 6ms late
            "duration_s": 90.0,
        })
        memory.log_session({
            "session_id": "sess_2",
            "beatmap_id": 12345,
            "beatmap_title": "Test Song [Insane]",
            "mode": "taiko",
            "result": "fail",
            "accuracy": 82.0,
            "combo": 40,
            "max_combo": 40,
            "miss_count": 4,
            "avg_timing_error_ms": 8.0,  # Hitting 8ms late
            "failure_reason": "hp_depleted",
            "duration_s": 25.0,
        })
        memory.log_session({
            "session_id": "sess_3",
            "beatmap_id": 99999,
            "beatmap_title": "Other Song [Easy]",
            "mode": "standard",
            "result": "pass",
            "accuracy": 100.0,
            "combo": 100,
            "max_combo": 100,
            "avg_timing_error_ms": -1.0,
        })

        # History retrieval
        history = memory.get_history(beatmap_id=12345)
        assert len(history) == 2
        assert history[0]["session_id"] == "sess_2"  # Newest first

        # Performance analysis for map 12345
        analysis = memory.analyze_performance(beatmap_id=12345)
        assert analysis["total_runs"] == 2
        assert analysis["pass_rate"] == 0.5
        assert analysis["avg_timing_error_ms"] == 7.0  # Average of 6.0 and 8.0
        # Since hitting 7ms late, recommendation should be positive lead adjustment (+min(6.0, 7.0 * 0.4) = +2.8ms)
        assert analysis["recommended_offset_adjustment_ms"] > 0.0
        assert "gõ trễ" in analysis["shiori_insight"]


def test_game_memory_self_tuning_application():
    """Verify apply_self_tuning adjusts the timing calibrator offset."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        mem_file = Path(tmp_dir) / "test_memory_tune.jsonl"
        memory = OsuGameMemory(memory_file_path=str(mem_file))

        # Reset calibrator baseline
        timing_calibrator.reset_offset()
        initial_offset = timing_calibrator.adaptive_offset_ms

        # Record a run with chronic late hitting (+10ms late)
        memory.log_session({
            "beatmap_id": 5555,
            "mode": "taiko",
            "result": "fail",
            "avg_timing_error_ms": 10.0,
            "accuracy": 75.0,
        })

        tuning = memory.apply_self_tuning(beatmap_id=5555, mode="taiko")
        assert tuning["applied_offset_ms"] > 0.0
        assert timing_calibrator.adaptive_offset_ms > initial_offset


def test_vts_bridge_emotion_mapping():
    """Verify OsuVTSBridge has slapstick, thinking, happy, and fail emotions defined."""
    assert "slapstick" in osu_vts_bridge.EMOTION_MAP
    assert "pan" in osu_vts_bridge.EMOTION_MAP
    assert "fail" in osu_vts_bridge.EMOTION_MAP
    assert "thinking" in osu_vts_bridge.EMOTION_MAP
    assert "happy" in osu_vts_bridge.EMOTION_MAP

    hk, exp = osu_vts_bridge.EMOTION_MAP["slapstick"]
    assert hk == "脸盆"


def test_hard_map_triggers_thinking_emotion():
    """Verify fast BPM beatmap triggers thinking emotion upon activation."""
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())

    captured_emotions = []

    def mock_emit(emotion="neutral", intensity=1.0, duration=3.0, text="", reason=""):
        captured_emotions.append((emotion, reason, text))

    app._emit_emotion_and_thought = mock_emit

    # Create dummy fast beatmap (BPM = 240)
    meta = MetadataSection(title="Camellia Fast Stream", artist="Camellia", version="Extra", beatmap_id=777)
    diff = DifficultySection(overall_difficulty=9.0)
    bm = BeatmapData(metadata=meta, difficulty=diff, mode=GameMode.TAIKO)
    bm.bpm = 240.0
    bm.total_length_ms = 60000.0

    app._active_beatmap = bm
    app._on_beatmap_activated(GameMode.TAIKO)

    hard_map_events = [e for e in captured_emotions if e[1] == "hard_map_alert"]
    assert len(hard_map_events) >= 1
    assert hard_map_events[0][0] == "thinking"
    assert "240 BPM" in hard_map_events[0][2]


def test_combo_milestones_and_slapstick_on_fail():
    """Verify combo 20, 50, 100 trigger happy/excited, and fail triggers slapstick (chảo úp đầu)."""
    app = OsuGameplayApp()
    app.controller.set_backend(MockInputBackend())

    captured_emotions = []

    def mock_emit(emotion="neutral", intensity=1.0, duration=3.0, text="", reason=""):
        captured_emotions.append((emotion, reason, text))

    app._emit_emotion_and_thought = mock_emit
    app.start(session_id="test_milestones_and_fail", run_background_thread=False)

    # 1. Combo 20
    app.feedback.metrics.current_combo = 20
    app.feedback.metrics.count_300 = 20
    app.step(simulated_time_ms=1000.0)
    c20 = [e for e in captured_emotions if e[1] == "combo_milestone" and "20 combo" in e[2]]
    assert len(c20) >= 1
    assert c20[0][0] == "happy"

    # 2. Combo 50
    app.feedback.metrics.current_combo = 50
    app.step(simulated_time_ms=2000.0)
    c50 = [e for e in captured_emotions if e[1] == "combo_milestone" and "50x COMBO" in e[2]]
    assert len(c50) >= 1
    assert c50[0][0] == "happy"

    # 3. Combo 100
    app.feedback.metrics.current_combo = 100
    app.step(simulated_time_ms=3000.0)
    c100 = [e for e in captured_emotions if e[1] == "combo_milestone" and "100x COMBO" in e[2]]
    assert len(c100) >= 1
    assert c100[0][0] == "excited"

    # 4. Fail / Miss burst -> Slapstick chảo úp đầu
    app.feedback.metrics.current_combo = 0
    app.feedback.metrics.count_miss = 4
    app._actions_dispatched_total = 10
    app.step(simulated_time_ms=4000.0)

    fail_events = [e for e in captured_emotions if e[1] == "fail_slapstick"]
    assert len(fail_events) >= 1
    assert fail_events[0][0] == "slapstick"
    assert "chảo" in fail_events[0][2]


def test_vts_bridge_lip_sync_and_motion_tracking():
    """Verify OsuVTSBridge supports lip-sync and dynamic facial/eye tracking."""
    import asyncio
    import threading
    from src.gameplay.osu.integrations.vts_bridge import OsuVTSBridge
    bridge = OsuVTSBridge()
    bridge.enabled = True

    # Test tracking coordinate mapping
    injected_params = []

    class MockClient:
        is_authenticated = True
        is_connected = True

        async def inject_parameters(self, params):
            injected_params.extend(params)
            return True

        async def inject_parameter(self, param, value, weight=1.0):
            injected_params.append({"id": param, "value": value, "weight": weight})
            return True

    bridge._client = MockClient()
    bridge._loop = asyncio.new_event_loop()
    t = threading.Thread(target=bridge._loop.run_forever, daemon=True)
    t.start()

    try:
        # Cursor at top-right (1400, 200) on 1600x900 screen
        bridge.update_gameplay_tracking(
            cursor_x=1400.0,
            cursor_y=200.0,
            window_w=1600.0,
            window_h=900.0,
            bpm=200.0,
            combo=35,
        )
        time.sleep(0.05)

        assert len(injected_params) >= 5
        param_map = {p["id"]: p["value"] for p in injected_params}
        assert param_map["FaceAngleX"] > 0.0  # Aiming to the right -> FaceAngleX > 0
        assert param_map["FaceAngleY"] > 0.0  # Aiming to top -> FaceAngleY > 0
        assert param_map["EyeLeftX"] > 0.0
        assert "BodyAngleX" in param_map
    finally:
        bridge._loop.call_soon_threadsafe(bridge._loop.stop)
        t.join(timeout=0.5)


def test_long_slider_preservation_and_mouse_lock():
    """Verify sliders longer than 3 seconds are preserved and mouse is locked until slider end."""
    from src.gameplay.osu.world.world_state import WorldStateManager
    from src.gameplay.osu.world.object_lifecycle import WorldObject, ObjectLifecycleState
    from src.gameplay.osu.beatmap.models import Slider, HitCircle
    from src.gameplay.osu.brains.standard_brain import StandardBrain

    wm = WorldStateManager()
    sl = Slider(
        x=200.0,
        y=200.0,
        time_ms=1000.0,
        hit_type=2,
        duration_ms=4000.0,  # 4 seconds long
        repeat_count=1,
    )
    w_obj = WorldObject(id="sl_long", hit_object=sl, lifecycle=ObjectLifecycleState.APPROACHING)
    wm.add_or_update(w_obj)

    # 1. At t=3000ms (slider is half-way through), prune_expired MUST NOT delete it!
    dropped = wm.prune_expired(current_time_ms=3000.0, max_age_ms=600.0)
    assert dropped == 0
    assert wm.get_object("sl_long") is not None

    # 2. StandardBrain must lock mouse until slider ends (t=5000ms)
    brain = StandardBrain()
    c2 = HitCircle(x=400.0, y=300.0, time_ms=5100.0, hit_type=1)
    w_c2 = WorldObject(id="c2", hit_object=c2, lifecycle=ObjectLifecycleState.APPROACHING)

    intents = brain.plan_actions(
        current_time_ms=2000.0,
        objects=[w_obj, w_c2],
        timeline_anchor_ns=100_000_000,
    )
    # Check that mouse movement for c2 does NOT start before slider ends (5000ms)
    c2_moves = [act for act in intents if "c2_move" in act.action_id]
    assert len(c2_moves) >= 1
    # Dispatch time for 5000ms is 100_000_000 + 5000 * 1_000_000 = 5_100_000_000ns
    for m in c2_moves:
        assert m.execute_at_ns >= (100_000_000 + int(5000.0 * 1_000_000) - 20_000_000), (
            f"Cursor left slider early: {m.execute_at_ns}"
        )

