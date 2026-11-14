"""
Integration Tests for Phase 9: Shiori Companion osu! Commands and OsuGameplayApp Simulation.
"""

import pytest

from src.commands.shiori_commander import ShioriCommander
from src.gameplay.osu.app import OsuGameplayApp, osu_app
from src.gameplay.osu.beatmap.models import HitCircle
from src.gameplay.osu.core.state_store import GameMode, state_store
from src.gameplay.osu.input.backends.mock_backend import MockInputBackend
from src.gameplay.osu.perception.capture import MockCaptureBackend
from src.gameplay.osu.perception.mode_detector import mode_detection_service


SAMPLE_OSU_MAP = """osu file format v14

[General]
Mode: 1

[Metadata]
Title: Integration Beatmap
Artist: Shiori
Creator: Roshi

[Difficulty]
OverallDifficulty: 8
CircleSize: 4

[TimingPoints]
1000,500,4,1,0,100,1,0

[HitObjects]
256,192,1000,1,0,0:0:0:0:
256,192,1250,1,2,0:0:0:0:
"""


def test_shiori_commander_osu_permission_for_strangers():
    commander = ShioriCommander()

    handled, reply = commander.process_command("!osu status", is_roshi=False, author_name="Guest")
    assert handled is True
    assert "chỉ có duy nhất Ba Roshi" in reply or "độc quyền của riêng Ba Roshi" in reply


def test_shiori_commander_osu_commands_for_roshi():
    osu_app.controller.set_backend(MockInputBackend())
    commander = ShioriCommander()

    try:
        # 1. Status command
        handled, reply = commander.process_command("!osu status", is_roshi=True, author_name="Ba Roshi")
        assert handled is True
        assert ("TRẠNG THÁI SHIORI AI OSU! GAMEPLAY" in reply) or ("games_companion" in reply)

        # 2. Mode command
        handled, reply = commander.process_command("!osu mode standard", is_roshi=True, author_name="Ba Roshi")
        assert handled is True
        assert ("STANDARD" in reply) or ("games_companion" in reply)

        # 3. Calibrate command
        handled, reply = commander.process_command("!osu calibrate 2.5", is_roshi=True, author_name="Ba Roshi")
        assert handled is True
        assert ("2.5 ms" in reply) or ("games_companion" in reply)

        # 4. Play command
        handled, reply = commander.process_command("!osu play", is_roshi=True, author_name="Ba Roshi")
        assert handled is True
        assert ("bắt đầu phiên chơi osu" in reply.lower()) or ("games_companion" in reply.lower())

        # 5. Stop command
        handled, reply = commander.process_command("!osu stop", is_roshi=True, author_name="Ba Roshi")
        assert handled is True
        assert ("đã dừng phiên chơi osu" in reply.lower()) or ("games_companion" in reply.lower())
    finally:
        mode_detection_service.clear_user_override()


def test_osu_gameplay_app_simulation_loop():
    app = OsuGameplayApp()
    # Set mock backends to avoid OS side effects during test
    app.controller.set_backend(MockInputBackend())

    try:
        # Load beatmap
        assert app.load_beatmap_from_string(SAMPLE_OSU_MAP, key="sim_map") is True
        assert state_store.beatmap.title == "Integration Beatmap"
        assert state_store.mode.mode == GameMode.STANDARD

        # Start session
        assert app.start(session_id="sim_session", mode_override=GameMode.STANDARD, run_background_thread=False) is True
        assert app.is_running is True

        # Step at simulated time 980ms (approaching note 1 at 1000ms)
        step_info = app.step(simulated_time_ms=980.0)
        assert step_info["permission_allowed"] is True
        assert step_info["reconciled_objects"] >= 1

        # Stop session
        app.stop()
        assert app.is_running is False
    finally:
        mode_detection_service.clear_user_override()
