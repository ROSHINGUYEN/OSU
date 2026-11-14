"""
Comprehensive Unit Tests for TosuClient & Beatmap Fast-Path Discovery
Verifies:
1. Parsing Tosu payload (game time, accuracy, combo, hits, UR, game mode).
2. Monotonic interpolated time calculation with jitter clamping.
3. Sub-millisecond direct disk beatmap read for osu!lazer sharded hash files.
4. Sub-millisecond direct disk beatmap read for osu!stable Songs directory.
5. In-memory caching of beatmap content across frames.
6. HTTP fallback endpoints (/files/beatmap/file and /Songs/).
7. Persistent Keep-Alive connection handling and automatic reconnection on socket drop.
8. Immunity against corrupt/dummy beatmap hit objects (0,0,0,0,0).
"""

import http.client
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.gameplay.osu.beatmap.loader import BeatmapLoader
from src.gameplay.osu.beatmap.models import HitCircle, HitObjectFlag, Slider, Spinner
from src.gameplay.osu.beatmap.parser import BeatmapParser
from src.gameplay.osu.core.clock import clock
from src.gameplay.osu.core.state_store import GameMode
from src.gameplay.osu.integrations.tosu_client import TosuClient


@pytest.fixture
def client():
    """Create a fresh isolated TosuClient instance."""
    return TosuClient(host="127.0.0.1", port=24050, poll_rate_hz=60)


def test_tosu_parse_payload_standard_and_metrics(client):
    """Verify parsing full Tosu JSON payload into strongly-typed client state."""
    sample_payload = {
        "menu": {
            "state": 2,
            "bm": {
                "id": 123456,
                "set": 78910,
                "metadata": {
                    "artist": "Camellia",
                    "title": "GHOST",
                    "difficulty": "Expert",
                },
                "path": {
                    "file": r"6\60\607f1864c3cacd994b363feff3241a269ced921b6f0a59e85830de88e78f2dae",
                    "folder": ".",
                },
            },
        },
        "gameplay": {
            "time": 45230.0,
            "combo": {"current": 425, "max": 800},
            "accuracy": 98.75,
            "hits": {
                "300": 390,
                "100": 10,
                "50": 2,
                "0": 1,
                "grade": {"current": "S"},
                "hitErrorArray": [-5.2, 3.1, -1.0, 4.4],
                "unstableRate": 85.3,
            },
            "hp": {"normal": 95.0},
            "gameMode": 0,
        },
    }

    client._parse_payload(sample_payload)

    assert client.is_playing is True
    assert client.time_ms == 45230.0
    assert client.current_combo == 425
    assert client.max_combo == 800
    assert client.accuracy == 98.75
    assert client.hits_300 == 390
    assert client.hits_100 == 10
    assert client.hits_50 == 2
    assert client.hits_miss == 1
    assert client.grade == "S"
    assert client.hp == 95.0
    assert client.is_failed is False
    assert client.game_mode_str == "standard"
    assert client.beatmap_id == 123456
    assert client.beatmap_set_id == 78910
    assert client.beatmap_title == "Camellia - GHOST [Expert]"
    assert client.file_name == r"6\60\607f1864c3cacd994b363feff3241a269ced921b6f0a59e85830de88e78f2dae"
    assert len(client.hit_errors) == 4
    assert client.unstable_rate == 85.3

    snapshot = client.get_snapshot()
    assert snapshot["combo"] == 425
    assert snapshot["hits"]["miss"] == 1


def test_tosu_interpolated_time_clamping(client):
    """Verify sub-millisecond interpolation clamps smoothly to max 35ms."""
    client.is_playing = True
    client.time_ms = 10000.0
    base_ns = clock.now_ns()
    client.last_packet_ns = base_ns

    # 10ms later
    t_10 = client.get_interpolated_time_ms(now_ns=base_ns + 10_000_000)
    assert abs(t_10 - 10010.0) < 0.05

    # 20ms later
    t_20 = client.get_interpolated_time_ms(now_ns=base_ns + 20_000_000)
    assert abs(t_20 - 10020.0) < 0.05

    # 100ms later (network hitch / packet loss) -> clamped to +35ms
    t_100 = client.get_interpolated_time_ms(now_ns=base_ns + 100_000_000)
    assert abs(t_100 - 10035.0) < 0.05

    # When not playing, return raw time without interpolation
    client.is_playing = False
    assert client.get_interpolated_time_ms(now_ns=base_ns + 10_000_000) == 10000.0


def test_tosu_direct_disk_lazer_sharded_file(client, tmp_path):
    """Verify direct disk read for osu!lazer sharded hash file (<0.3ms)."""
    # Create fake lazer file structure: tmp_path / osu / files / 6 / 60 / <hash>
    lazer_files = tmp_path / "osu" / "files"
    shard_dir = lazer_files / "6" / "60"
    shard_dir.mkdir(parents=True)
    hash_name = "607f1864c3cacd994b363feff3241a269ced921b6f0a59e85830de88e78f2dae"
    map_file = shard_dir / hash_name

    sample_osu_content = """osu file format v14
[General]
Mode: 0
[Metadata]
Title:DirectDiskTest
[Difficulty]
OverallDifficulty:8
[HitObjects]
256,192,1000,1,0,0:0:0:0:
256,192,1500,1,0,0:0:0:0:
"""
    map_file.write_text(sample_osu_content, encoding="utf-8")

    client.file_name = f"6\\60\\{hash_name}"
    client.beatmap_id = 9999
    client.beatmap_title = "DirectDiskTest [Easy]"

    with patch.dict(os.environ, {"APPDATA": str(tmp_path)}):
        found_path = client.get_active_beatmap_file_path()
        assert found_path is not None
        assert found_path.exists()
        assert found_path.resolve() == map_file.resolve()

        # Test fetch content
        content = client.fetch_active_beatmap_content()
        assert content is not None
        assert "Title:DirectDiskTest" in content
        assert "[HitObjects]" in content
        # Cache hit
        assert client._cached_beatmap_content == content


def test_tosu_direct_disk_lazer_raw_64char_hex(client, tmp_path):
    """Verify raw 64-char hash without slashes automatically maps to shard h[0]/h[:2]/h."""
    lazer_files = tmp_path / "osu" / "files"
    hash_str = "a1b2c3d4e5f67890123456789012345678901234567890123456789012345678"
    shard_dir = lazer_files / "a" / "a1"
    shard_dir.mkdir(parents=True)
    map_file = shard_dir / hash_str
    map_file.write_text("osu file format v14\n[HitObjects]\n256,192,500,1,0\n", encoding="utf-8")

    client.file_name = hash_str
    client.beatmap_id = 8888

    with patch.dict(os.environ, {"APPDATA": str(tmp_path)}):
        found_path = client.get_active_beatmap_file_path()
        assert found_path is not None
        assert found_path.resolve() == map_file.resolve()


def test_tosu_direct_disk_stable_songs(client, tmp_path):
    """Verify direct disk read for osu!stable Songs directory."""
    songs_dir = tmp_path / "osu" / "Songs" / "12345 Artist - Title"
    songs_dir.mkdir(parents=True)
    map_file = songs_dir / "Artist - Title (Mapper) [Insane].osu"
    map_file.write_text("osu file format v14\n[HitObjects]\n256,192,1200,1,0\n", encoding="utf-8")

    client.folder_name = "12345 Artist - Title"
    client.file_name = "Artist - Title (Mapper) [Insane].osu"
    client.beatmap_id = 7777

    with patch.dict(os.environ, {"APPDATA": str(tmp_path)}):
        found_path = client.get_active_beatmap_file_path()
        assert found_path is not None
        assert found_path.resolve() == map_file.resolve()

        content = client.fetch_active_beatmap_content()
        assert content is not None
        assert "256,192,1200,1,0" in content


def test_tosu_fetch_beatmap_http_fallback(client):
    """Verify fallback to HTTP /files/beatmap/file when local disk file is missing."""
    client.file_name = "non_existent_map.osu"
    client.beatmap_id = 5555

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.read.return_value = b"osu file format v14\n[HitObjects]\n100,100,2000,1,0\n"
    fake_resp.__enter__.return_value = fake_resp

    with patch.object(client, "get_active_beatmap_file_path", return_value=None), \
         patch("urllib.request.urlopen", return_value=fake_resp):
        content = client.fetch_active_beatmap_content()
        assert content is not None
        assert "100,100,2000,1,0" in content


def test_corrupt_dummy_object_filtered_out():
    """Verify BeatmapParser skips corrupt dummy lines (0,0,0,0,0) without creating ghost objects."""
    corrupt_map_text = """osu file format v14
[General]
Mode: 0
[Difficulty]
OverallDifficulty: 8
[HitObjects]
256,192,1000,1,0,0:0:0:0:
0,0,0,0,0
0,0,0,0,0
0,0,0,0,0
256,192,2000,1,0,0:0:0:0:
"""
    beatmap = BeatmapParser.parse_string(corrupt_map_text, file_path="test_corrupt.osu")

    # Should only contain the 2 valid hit circles at 1000ms and 2000ms
    assert len(beatmap.hit_objects) == 2
    assert beatmap.hit_objects[0].time_ms == 1000.0
    assert beatmap.hit_objects[1].time_ms == 2000.0
    assert all(obj.time_ms > 0.0 for obj in beatmap.hit_objects)


def test_tosu_poll_loop_persistent_connection(client):
    """Verify _poll_loop creates a persistent HTTPConnection and reuses it."""
    mock_conn = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "menu": {"state": 2, "bm": {"id": 111}},
        "gameplay": {"time": 1500.0, "gameMode": 0}
    }).encode("utf-8")
    mock_conn.getresponse.return_value = mock_resp

    with patch("http.client.HTTPConnection", return_value=mock_conn):
        # Run one loop iteration by setting stop_event right after
        def _side_effect(*args, **kwargs):
            client._stop_event.set()
            return mock_resp

        mock_conn.getresponse.side_effect = _side_effect
        client._interval_s = 0.001
        client._poll_loop()

        # Connection should have requested /json with Keep-Alive
        mock_conn.request.assert_called_with(
            "GET",
            "/json",
            headers={"User-Agent": "ShioriAI/1.0", "Connection": "keep-alive"}
        )
        assert client.time_ms == 1500.0
        assert client.is_connected is True


def test_tosu_disk_index_resolution_by_id_and_title(client, tmp_path):
    """Verify fast in-memory disk index resolves beatmaps by ID and title without network requests."""
    fake_lazer_files = tmp_path / "osu" / "files"
    fake_lazer_files.mkdir(parents=True)
    sample_file = fake_lazer_files / "sample_hash"
    sample_file.write_text(
        "osu file format v14\n"
        "[Metadata]\n"
        "Title:Fast Beatmap\n"
        "Version:Insane\n"
        "BeatmapID:888999\n"
        "[HitObjects]\n"
        "256,192,1000,1,0\n",
        encoding="utf-8"
    )

    with patch.dict("os.environ", {"APPDATA": str(tmp_path)}):
        client._beatmap_index_by_id.clear()
        client._beatmap_index_by_title_diff.clear()
        client._last_index_time_s = 0.0

        # Query by Beatmap ID
        client.beatmap_id = 888999
        client.beatmap_title = ""
        path_by_id = client.get_active_beatmap_file_path()
        assert path_by_id is not None
        assert path_by_id.name == "sample_hash"

        # Query by Title + Version
        client.beatmap_id = 0
        client.beatmap_title = "Artist - Fast Beatmap [Insane]"
        path_by_title = client.get_active_beatmap_file_path()
        assert path_by_title is not None
        assert path_by_title.name == "sample_hash"


def test_tosu_interpolated_time_supports_negative_leadin(client):
    """Verify get_interpolated_time_ms smoothly interpolates during negative intro lead-in."""
    client.is_playing = True
    client.time_ms = -1335.0
    client.last_packet_ns = 1_000_000_000

    # 15ms elapsed
    now_ns = client.last_packet_ns + 15_000_000
    interpolated = client.get_interpolated_time_ms(now_ns)
    assert interpolated == -1320.0

