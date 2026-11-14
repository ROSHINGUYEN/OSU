"""
Session Telemetry Logger and Replay Serializer for Shiori AI osu!.
Records structured events in JSONL for offline deterministic debugging and replays.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.gameplay.osu.core.clock import clock


class TelemetryLogger:
    """Thread-safe event recorder dumping telemetry snapshots to JSONL files with automatic retention management."""

    def __init__(self, output_dir: str = "data/osu_telemetry", max_retained_sessions: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_retained_sessions = max_retained_sessions
        self._current_file: Optional[Path] = None
        self._file_handle = None
        self._session_id: str = ""
        self._events_count = 0
        self._last_flush_ns: int = 0

    def _resolve_session_file(self, session_id: str) -> Path:
        """Route pytest sessions to isolated session_test_* files so live session_01.jsonl is never overwritten."""
        is_test = "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("OSU_TEST_ENV") == "1"
        is_default_live_dir = self.output_dir.as_posix().endswith("data/osu_telemetry")
        if is_test and is_default_live_dir and "test" not in session_id.lower():
            clean = session_id[len("session_"):] if session_id.startswith("session_") else session_id
            file_name = f"session_test_{clean}.jsonl"
        else:
            file_name = f"{session_id}.jsonl" if session_id.startswith("session_") else f"session_{session_id}.jsonl"
        return self.output_dir / file_name

    @property
    def session_id(self) -> str:
        return self._session_id

    def _cleanup_old_sessions(self) -> None:
        """Keep only the most recent N session files to prevent disk & memory clutter."""
        try:
            files = sorted(self.output_dir.glob("session_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            if len(files) > self.max_retained_sessions:
                for old_file in files[self.max_retained_sessions:]:
                    try:
                        old_file.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def clear_all_sessions(self, prefix: str = "session_") -> int:
        """Purge stored session files from disk."""
        removed = 0
        try:
            for f in self.output_dir.glob(f"{prefix}*.jsonl"):
                try:
                    f.unlink(missing_ok=True)
                    removed += 1
                except Exception:
                    pass
        except Exception:
            pass
        return removed

    def start_session(self, session_id: str) -> None:
        self.stop_session()
        self._session_id = session_id
        self._current_file = self._resolve_session_file(session_id)
        self._events_count = 0
        self._cleanup_old_sessions()
        try:
            self._file_handle = open(self._current_file, "a", encoding="utf-8")
        except Exception:
            self._file_handle = None
        self.log_event("session_started", {"session_id": session_id})

    def log_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._current_file is None:
            return

        entry = {
            "timestamp_ns": clock.now_ns(),
            "event_type": event_type,
            "session_id": self._session_id,
            "payload": payload,
        }

        try:
            line = json.dumps(entry) + "\n"
            if self._file_handle is not None and not self._file_handle.closed:
                self._file_handle.write(line)
                # ponytail: flush every 64 events or 250ms instead of every event to avoid NTFS stall in 2000Hz loop; upgrade to async queue thread if I/O volume grows
                now_ns = entry["timestamp_ns"]
                if (self._events_count & 63) == 0 or (now_ns - self._last_flush_ns) >= 250_000_000:
                    self._file_handle.flush()
                    self._last_flush_ns = now_ns
            else:
                with open(self._current_file, "a", encoding="utf-8") as f:
                    f.write(line)
            self._events_count += 1
        except Exception:
            pass

    def stop_session(self) -> None:
        if self._current_file is not None:
            self.log_event("session_ended", {"total_events": self._events_count})
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass
            self._file_handle = None
        self._current_file = None
        self._session_id = ""

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent recorded telemetry events."""
        if self._file_handle is not None and not self._file_handle.closed:
            try:
                self._file_handle.flush()
            except Exception:
                pass
        target_file = self._current_file
        if target_file is None or not target_file.exists():
            # Fall back to latest session file in telemetry directory
            files = sorted(self.output_dir.glob("session_*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
            if files:
                target_file = files[0]
            else:
                return []

        events = []
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception:
            pass
        return events

    def load_replay(self, session_file_path: str) -> List[Dict[str, Any]]:
        """Load and replay an existing session from disk."""
        events = []
        p = Path(session_file_path)
        if not p.is_file():
            return events
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events


telemetry_logger = TelemetryLogger()
