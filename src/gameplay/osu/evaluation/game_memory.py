"""
Persistent Game Memory and Self-Tuning System for Shiori AI osu!
================================================================
Records every gameplay outcome (win, fail, quit/abort) to JSONL, analyzes historical
timing errors and miss patterns, and provides adaptive fine-tuning (lead offsets,
difficulty anticipation, and companion reflection commentary) for future plays.
"""

import datetime
import json
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Shiori.Osu.Memory")


class OsuGameMemory:
    """Thread-safe persistent game history, 1D Kalman beatmap profiling, and skill memory for Shiori osu!."""

    def __init__(self, memory_file_path: Optional[str] = None):
        if memory_file_path:
            self._explicit_file: Optional[Path] = Path(memory_file_path)
        else:
            self._explicit_file = None
        self._root_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
        self._lock = threading.RLock()
        self._profiles_cache: Optional[Dict[str, Any]] = None
        self._profiles_cache_mtime: float = 0.0
        self._skills_cache: Optional[Dict[str, Any]] = None
        self._skills_cache_mtime: float = 0.0

    @property
    def memory_file(self) -> Path:
        """Level A: JSONL session log store."""
        if self._explicit_file:
            target = self._explicit_file
        elif "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("OSU_TEST_ENV") == "1":
            target = self._root_dir / "data" / "osu_test_memory.jsonl"
        else:
            target = self._root_dir / "data" / "osu_gameplay_memory.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def profile_file(self) -> Path:
        """Level B: JSON store for persistent beatmap profiles (Kalman fine-tuning)."""
        if "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("OSU_TEST_ENV") == "1":
            target = self._root_dir / "data" / "osu_test_profiles.json"
        else:
            target = self._root_dir / "data" / "osu_beatmap_profiles.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def skill_file(self) -> Path:
        """Level C: JSON store for cross-beatmap mechanic/skill proficiencies."""
        if "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("OSU_TEST_ENV") == "1":
            target = self._root_dir / "data" / "osu_test_skill_memory.json"
        else:
            target = self._root_dir / "data" / "osu_skill_memory.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _load_profiles(self) -> Dict[str, Any]:
        """Load beatmap profiles from disk with in-memory mtime caching."""
        p_file = self.profile_file
        if not p_file.exists():
            return {}
        try:
            mtime = p_file.stat().st_mtime
            if self._profiles_cache is not None and mtime == self._profiles_cache_mtime:
                return self._profiles_cache
            with open(p_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._profiles_cache = data
            self._profiles_cache_mtime = mtime
            return data
        except Exception as e:
            logger.debug(f"Failed to load beatmap profiles: {e}")
            return self._profiles_cache or {}

    def _load_skills(self) -> Dict[str, Any]:
        """Load skill memory from disk with in-memory mtime caching."""
        s_file = self.skill_file
        if not s_file.exists():
            return {}
        try:
            mtime = s_file.stat().st_mtime
            if self._skills_cache is not None and mtime == self._skills_cache_mtime:
                return self._skills_cache
            with open(s_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._skills_cache = data
            self._skills_cache_mtime = mtime
            return data
        except Exception as e:
            logger.debug(f"Failed to load skill memory: {e}")
            return self._skills_cache or {}

    def get_beatmap_profile(
        self,
        beatmap_id: int,
        title: str = "",
        mode: str = "standard",
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve fine-tuned machine learning profile for a specific beatmap.
        Supports mode isolation with fallback to legacy un-prefixed keys.
        """
        mode_str = str(mode or "standard").lower()
        if mode_str in ("unknown", "none", ""):
            mode_str = "standard"
        with self._lock:
            data = self._load_profiles()
            # Primary mode-isolated key
            primary_key = f"{mode_str}:{beatmap_id}" if beatmap_id > 0 else f"{mode_str}:title_{title.strip().lower()}"
            prof = data.get(primary_key)
            if prof is not None:
                return dict(prof)

            # Fallback to legacy un-prefixed key
            legacy_key = str(beatmap_id) if beatmap_id > 0 else f"title_{title.strip().lower()}"
            legacy_prof = data.get(legacy_key)
            return dict(legacy_prof) if legacy_prof is not None else None

    def update_beatmap_profile_kalman(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bayesian Machine Learning Update (1D Kalman Filter) for beatmap-specific lead-time optimization.
        Estimates true physical latency offset from noisy in-game hit errors.
        Guarded against 0-hit sessions, quits, and unverified simulations.
        """
        bm_id = int(record.get("beatmap_id") or 0)
        title = str(record.get("beatmap_title") or "Unknown").strip()
        mode_str = str(record.get("mode") or "standard").lower()
        if mode_str in ("unknown", "none", ""):
            mode_str = "standard"
        key = f"{mode_str}:{bm_id}" if bm_id > 0 else f"{mode_str}:title_{title.lower()}"

        with self._lock:
            profiles = dict(self._load_profiles())

            # Legacy fallback migration
            legacy_key = str(bm_id) if bm_id > 0 else f"title_{title.lower()}"
            if key not in profiles and legacy_key in profiles:
                profiles[key] = profiles.pop(legacy_key)

            prof = dict(profiles.get(key, {
                "beatmap_id": bm_id,
                "title": title,
                "mode": mode_str,
                "total_plays": 0,
                "total_quits": 0,
                "best_accuracy": 0.0,
                "learned_adaptive_offset_ms": 0.0,
                "kalman_variance": 4.0,  # Initial prior uncertainty
                "archetype": "balanced",
                "last_avg_error_ms": 0.0,
                "updated_at": "",
            }))

            # 1. Quality Gates: Reject incomplete or unrepresentative plays
            has_duration = "duration_s" in record
            duration_s = float(record.get("duration_s", 0.0))
            hits_dict = record.get("hits", {})
            has_hits = bool(hits_dict) or ("combo" in record) or ("miss_count" in record)
            total_hits = sum(hits_dict.values()) if hits_dict else (
                record.get("combo", 0) + record.get("miss_count", 0)
            )

            is_early_quit = (
                (has_hits and total_hits < 5)
                or (has_duration and duration_s < 2.0)
                or (record.get("result") == "quit" and not has_hits and float(record.get("accuracy", 0.0)) == 0.0)
            )

            if is_early_quit:
                prof["total_quits"] = prof.get("total_quits", 0) + 1
                profiles[key] = prof
                try:
                    with open(self.profile_file, "w", encoding="utf-8") as f:
                        json.dump(profiles, f, ensure_ascii=False, indent=2)
                    self._profiles_cache = profiles
                    self._profiles_cache_mtime = self.profile_file.stat().st_mtime
                except Exception:
                    pass
                return dict(prof)

            # 2. Extract observation & noise properties
            observed_err = float(record.get("avg_timing_error_ms") or 0.0)
            acc = float(record.get("accuracy") or 0.0)
            timing_samples_count = len(record.get("timing_errors", [])) or total_hits
            raw_variance = float(record.get("timing_error_variance", 4.0))

            # 3. 1D Bayesian Kalman Filter Mathematics
            x_prev = float(prof.get("learned_adaptive_offset_ms", 0.0))
            p_prev = float(prof.get("kalman_variance", 4.0))

            # Only update if timing error is within sane physiological limits (|err| <= 35ms)
            if abs(observed_err) <= 35.0:
                # Process noise Q (slight drift allowance per session)
                q_process = 0.08
                p_prior = p_prev + q_process

                # Measurement noise R = sigma^2 / N + accuracy penalty
                n_samples = max(1, min(100, timing_samples_count))
                accuracy_penalty = max(0.0, (100.0 - acc) / 25.0)
                r_meas = max(0.5, (raw_variance / n_samples) + accuracy_penalty)

                # Kalman gain
                k_gain = p_prior / (p_prior + r_meas)

                # State & Covariance updates
                x_new = x_prev + k_gain * observed_err
                p_new = (1.0 - k_gain) * p_prior

                # Strict bounded clamping to prevent drift
                prof["learned_adaptive_offset_ms"] = round(max(-6.0, min(6.0, x_new)), 2)
                prof["kalman_variance"] = round(max(0.1, min(8.0, p_new)), 3)

            prof["total_plays"] = prof.get("total_plays", 0) + 1
            prof["best_accuracy"] = max(prof.get("best_accuracy", 0.0), acc)
            prof["last_avg_error_ms"] = round(observed_err, 2)
            prof["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            # Mechanic Archetype Classification
            if mode_str == "taiko":
                prof["archetype"] = "rhythm_percussion"
            elif mode_str == "catch":
                prof["archetype"] = "catch_hyperdash"
            elif mode_str == "mania":
                prof["archetype"] = "chord_alternation"
            elif record.get("is_stream_heavy"):
                prof["archetype"] = "stream_speed"
            else:
                prof["archetype"] = "jump_aim"

            profiles[key] = prof

            try:
                with open(self.profile_file, "w", encoding="utf-8") as f:
                    json.dump(profiles, f, ensure_ascii=False, indent=2)
                self._profiles_cache = profiles
                self._profiles_cache_mtime = self.profile_file.stat().st_mtime
                logger.info(
                    f"[Shiori.ML.Profile] Updated Kalman ML Profile for '{title}' ({mode_str}): "
                    f"Learned Offset: {prof['learned_adaptive_offset_ms']:+.2f}ms | Best Acc: {prof['best_accuracy']}%"
                )
            except Exception as e:
                logger.warning(f"Failed to persist beatmap profile: {e}")

        return dict(prof)

    def record_skill_performance(
        self,
        pattern_name: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Level C Skill Memory: Records performance for specific mechanics across all beatmaps.
        (e.g., 'stream_speed', 'jump_aim', 'slider_flow', 'finger_alternation').
        """
        with self._lock:
            skills = dict(self._load_skills())
            pattern_key = str(pattern_name).strip().lower()

            skill = dict(skills.get(pattern_key, {
                "pattern": pattern_key,
                "samples_count": 0,
                "avg_accuracy": 0.0,
                "proficiency_score": 0.5,  # Baseline 50%
                "timing_bias_ms": 0.0,
                "updated_at": "",
            }))

            acc = float(metrics.get("accuracy", 0.0))
            err_ms = float(metrics.get("avg_timing_error_ms", 0.0))
            count = skill.get("samples_count", 0)

            # Exponential Moving Average (alpha = 0.15)
            alpha = 0.15 if count > 0 else 1.0
            new_acc = (1.0 - alpha) * skill.get("avg_accuracy", 0.0) + alpha * acc
            new_bias = (1.0 - alpha) * skill.get("timing_bias_ms", 0.0) + alpha * err_ms

            # Proficiency score computed from accuracy and tight timing
            score = max(0.0, min(1.0, (new_acc / 100.0) * max(0.2, 1.0 - abs(new_bias) / 20.0)))

            skill["samples_count"] = count + 1
            skill["avg_accuracy"] = round(new_acc, 2)
            skill["timing_bias_ms"] = round(new_bias, 2)
            skill["proficiency_score"] = round(score, 3)
            skill["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

            skills[pattern_key] = skill

            try:
                with open(self.skill_file, "w", encoding="utf-8") as f:
                    json.dump(skills, f, ensure_ascii=False, indent=2)
                self._skills_cache = skills
                self._skills_cache_mtime = self.skill_file.stat().st_mtime
            except Exception as e:
                logger.warning(f"Failed to persist skill memory: {e}")

        return dict(skill)

    def get_skill_profile(self, pattern_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve cross-beatmap mechanic profile."""
        with self._lock:
            skills = self._load_skills()
            prof = skills.get(str(pattern_name).strip().lower())
            return dict(prof) if prof is not None else None

    def get_all_skills(self) -> Dict[str, Any]:
        """Retrieve all recorded skill proficiencies."""
        with self._lock:
            return dict(self._load_skills())

    def log_session(self, record: Dict[str, Any]) -> None:
        """
        Record a gameplay session (pass, fail, quit) to persistent memory JSONL.
        """
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        raw_mode = str(record.get("mode") or "standard").lower()
        norm_mode = "standard" if raw_mode in ("unknown", "none", "") else raw_mode
        prev_prof = self.get_beatmap_profile(int(record.get("beatmap_id") or 0), str(record.get("beatmap_title") or ""), norm_mode)
        prev_best_acc = float(prev_prof.get("best_accuracy", 0.0)) if prev_prof else 0.0
        entry = {
            "timestamp": record.get("timestamp") or now_iso,
            "session_id": record.get("session_id", "session_unknown"),
            "beatmap_id": int(record.get("beatmap_id") or 0),
            "beatmap_title": str(record.get("beatmap_title") or "Unknown Beatmap"),
            "artist": str(record.get("artist") or "Unknown Artist"),
            "version": str(record.get("version") or "Normal"),
            "mode": norm_mode,
            "result": str(record.get("result") or "quit").lower(),  # pass | fail | quit
            "accuracy": round(float(record.get("accuracy") or 0.0), 2),
            "combo": int(record.get("combo") or 0),
            "max_combo": int(record.get("max_combo") or 0),
            "miss_count": int(record.get("miss_count") or 0),
            "hits": dict(record.get("hits") or {}),
            "avg_timing_error_ms": round(float(record.get("avg_timing_error_ms") or 0.0), 2),
            "timing_error_variance": round(float(record.get("timing_error_variance") or 0.0), 2),
            "timing_errors": [round(float(x), 2) for x in list(record.get("timing_errors") or [])[-64:]],
            "provenance": str(record.get("provenance") or "unknown"),
            "failure_reason": str(record.get("failure_reason") or "normal_exit"),
            "duration_s": round(float(record.get("duration_s") or 0.0), 2),
            "tuning_applied": dict(record.get("tuning_applied") or {}),
            "diagnostics": record.get("diagnostics"),
        }

        # 1. Closed-loop autonomous diagnosis & adaptive policy fine-tuning
        if not entry["diagnostics"]:
            try:
                from src.gameplay.osu.evaluation.diagnostics import diagnostics_engine
                from src.gameplay.osu.planning.adaptive_policy import adaptive_policy

                diag_report = diagnostics_engine.diagnose_session(entry)
                entry["diagnostics"] = diag_report.to_dict()
                total_hits_val = sum(entry["hits"].values()) if entry["hits"] else (entry["combo"] + entry["miss_count"])
                is_valid_run = entry["result"] != "quit" and (total_hits_val >= 5 or entry["duration_s"] >= 5.0 or not entry["hits"])
                if is_valid_run:
                    if prev_best_acc > 0.0 and entry["result"] == "pass":
                        adaptive_policy.check_and_rollback_regression(entry["accuracy"], prev_best_acc)
                    adaptive_policy.update_from_diagnosis(diag_report, session_metrics=entry)
            except Exception as diag_err:
                logger.debug(f"Failed to auto-diagnose session: {diag_err}")

        with self._lock:
            try:
                line = json.dumps(entry, ensure_ascii=False) + "\n"
                with open(self.memory_file, "a", encoding="utf-8") as f:
                    f.write(line)
                logger.info(
                    f"Logged game memory for '{entry['beatmap_title']}' ({entry['mode']}) -> "
                    f"Result: {entry['result'].upper()} | Acc: {entry['accuracy']}% | MaxCombo: {entry['max_combo']}x"
                )
                # Auto-train Bayesian Kalman model with this new session result
                self.update_beatmap_profile_kalman(record)

                # Record Level C skill performance if mechanic archetype is present
                archetype = record.get("archetype") or ("stream_speed" if record.get("is_stream_heavy") else "jump_aim")
                self.record_skill_performance(archetype, {
                    "accuracy": entry["accuracy"],
                    "avg_timing_error_ms": entry["avg_timing_error_ms"],
                })
            except Exception as e:
                logger.warning(f"Failed to write game memory to {self.memory_file}: {e}")

    def get_history(
        self,
        beatmap_id: Optional[int] = None,
        mode: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve past gameplay records matching beatmap_id or mode, newest first."""
        records: List[Dict[str, Any]] = []
        if not self.memory_file.exists():
            return records

        with self._lock:
            try:
                with open(self.memory_file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if beatmap_id is not None and beatmap_id > 0:
                                if rec.get("beatmap_id") != beatmap_id:
                                    continue
                            if mode is not None:
                                if rec.get("mode", "").lower() != mode.lower():
                                    continue
                            records.append(rec)
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Failed to read game memory from {self.memory_file}: {e}")

        records.reverse()  # Newest first
        return records[:limit]

    def analyze_performance(
        self,
        beatmap_id: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze recent historical attempts to determine timing trends and self-tuning recommendations.
        """
        history = self.get_history(beatmap_id=beatmap_id, mode=mode, limit=15)
        if not history:
            return {
                "total_runs": 0,
                "pass_rate": 0.0,
                "avg_accuracy": 0.0,
                "avg_timing_error_ms": 0.0,
                "recommended_offset_adjustment_ms": 0.0,
                "is_difficult": False,
                "shiori_insight": "",
            }

        total_runs = len(history)
        passes = sum(1 for r in history if r.get("result") == "pass")
        fails = sum(1 for r in history if r.get("result") == "fail")
        pass_rate = passes / total_runs

        accuracies = [r.get("accuracy", 0.0) for r in history if r.get("accuracy", 0.0) > 0.0]
        avg_acc = (sum(accuracies) / len(accuracies)) if accuracies else 0.0

        timing_errors = [
            r.get("avg_timing_error_ms", 0.0)
            for r in history
            if abs(r.get("avg_timing_error_ms", 0.0)) > 0.1
        ]
        avg_err = (sum(timing_errors) / len(timing_errors)) if timing_errors else 0.0

        rec_adj = 0.0
        if avg_err > 2.0:
            rec_adj = min(6.0, round(avg_err * 0.45, 1))
        elif avg_err < -2.0:
            rec_adj = max(-6.0, round(avg_err * 0.45, 1))

        is_difficult = (fails / total_runs >= 0.4) or (avg_acc < 85.0 and total_runs >= 2)

        # Generate Shiori's internal companion insight
        insight = ""
        if rec_adj > 0:
            insight = (
                f"Shiori nhớ màn này rồi Ba! Lần trước con bị gõ trễ ({avg_err:+.1f}ms) "
                f"nên lần này con đã tự bù thêm +{rec_adj:.1f}ms lead-time để bắt nhịp chuẩn xác hơn nè! ♡"
            )
        elif rec_adj < 0:
            insight = (
                f"Lần trước con gõ hơi sớm ({avg_err:+.1f}ms) nên Shiori đã tự điều chỉnh trễ lại {rec_adj:.1f}ms "
                f"cho khớp tiếng trống bài hát nha Ba! ♡"
            )
        elif is_difficult:
            insight = (
                f"Màn này trong ký ức của con có tiết tấu rất gắt (tỷ lệ trúng {avg_acc:.1f}%)! "
                f"Lần này con sẽ tập trung cao độ 100% để phục thù nè Ba! 💭🔥"
            )
        elif total_runs > 0:
            insight = (
                f"Màn quen thuộc của Ba con mình nè! Thành tích gần nhất: {avg_acc:.1f}% accuracy! "
                f"Shiori sẵn sàng tạo kỷ lục mới rồi ạ! ✨"
            )

        return {
            "total_runs": total_runs,
            "pass_rate": round(pass_rate, 2),
            "avg_accuracy": round(avg_acc, 2),
            "avg_timing_error_ms": round(avg_err, 2),
            "recommended_offset_adjustment_ms": rec_adj,
            "is_difficult": is_difficult,
            "shiori_insight": insight,
        }

    def apply_self_tuning(
        self,
        beatmap_id: Optional[int] = None,
        mode: Optional[str] = None,
        title: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze past runs, query persistent Kalman ML profile, and safely apply recommended
        adaptive offset to the system calibrator (preserving hardware baseline).
        """
        from src.gameplay.osu.timing.calibration import timing_calibrator

        # 1. Reset dynamic adaptive offset but preserve hardware baseline compensation
        timing_calibrator.reset_offset(reset_baseline=False)

        # 2. Check for specialized Kalman ML profile first
        bm_id_val = int(beatmap_id or 0)
        mode_val = str(mode or "standard").lower()
        if mode_val in ("unknown", "none", ""):
            mode_val = "standard"
        prof = self.get_beatmap_profile(beatmap_id=bm_id_val, title=title, mode=mode_val)

        if prof and prof.get("total_plays", 0) > 0:
            learned_adj = float(prof.get("learned_adaptive_offset_ms", 0.0))
            clamped_adj = max(-6.0, min(6.0, learned_adj))
            new_total = timing_calibrator.set_adaptive_offset_ms(clamped_adj)
            archetype = prof.get("archetype", "balanced")
            plays = prof.get("total_plays", 1)
            best_acc = prof.get("best_accuracy", 0.0)
            try:
                from src.gameplay.osu.evaluation.strategy_learner import strategy_learner
                from src.gameplay.osu.planning.adaptive_policy import adaptive_policy
                skill_prof = self.get_skill_profile(archetype)
                strat = strategy_learner.recommend_strategy(archetype, skill_prof)
                for sk in ("aim_smoothing_steps", "preparation_lead_ms", "deadline_margin_ms"):
                    if sk in strat:
                        adaptive_policy.params[sk] = strat[sk]
            except Exception:
                strat = {}

            insight = (
                f"Shiori đã nạp mô hình học máy Kalman cho bài '{prof.get('title', title or 'này')}': "
                f"Nhận diện cơ chế {archetype}, tự động bù trừ {clamped_adj:+.1f}ms "
                f"(Kỷ lục cũ: {best_acc:.1f}% qua {plays} lượt chơi)! Shiori tự tin 100% nha Ba Roshi! ♡"
            )

            logger.info(
                f"[Shiori.Osu.Memory] Loaded Kalman ML profile for ID={bm_id_val} ({mode_val}): "
                f"Offset {clamped_adj:+.2f}ms (Total Lead: {new_total:.1f}ms) | {archetype}"
            )

            return {
                "total_runs": plays,
                "best_accuracy": best_acc,
                "applied_offset_ms": clamped_adj,
                "archetype": archetype,
                "strategy": strat,
                "shiori_insight": insight,
            }

        # 3. Fallback to statistical JSONL history analysis
        analysis = self.analyze_performance(beatmap_id=beatmap_id, mode=mode)
        rec_adj = analysis.get("recommended_offset_adjustment_ms", 0.0)

        applied_offset = 0.0
        if abs(rec_adj) >= 0.5:
            try:
                clamped_adj = max(-6.0, min(6.0, rec_adj))
                new_total = timing_calibrator.set_adaptive_offset_ms(clamped_adj)
                applied_offset = clamped_adj
                logger.info(
                    f"[Shiori.Osu.Memory] Fine-tuned timing from game memory: offset set to {clamped_adj:+.1f}ms "
                    f"(Total Lead: {new_total:.1f}ms) based on {analysis['total_runs']} previous runs."
                )
            except Exception as e:
                logger.warning(f"Could not apply memory fine-tuning to calibrator: {e}")

        analysis["applied_offset_ms"] = applied_offset
        return analysis


osu_game_memory = OsuGameMemory()
