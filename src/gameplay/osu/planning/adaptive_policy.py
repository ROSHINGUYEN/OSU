"""
Adaptive Planning Policy for Shiori AI osu! Planning Engine.
=============================================================
Manages dynamic, bounded planning hyperparameters (aim smoothing, slider waypoint
density, deadline margin, preparation lead time) based on diagnosis feedback,
with automatic regression detection and rollback mechanisms.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.gameplay.osu.evaluation.diagnostics import DiagnosisReport, ErrorCategory


@dataclass
class PolicyBounds:
    """Safe physiological and mechanical parameter boundaries."""
    min_lead_offset_ms: float = -6.0
    max_lead_offset_ms: float = 6.0

    min_prep_lead_ms: float = 4.0
    max_prep_lead_ms: float = 16.0

    min_slider_density: int = 8
    max_slider_density: int = 24

    min_aim_smoothing_steps: int = 1
    max_aim_smoothing_steps: int = 6

    min_deadline_margin_ms: float = 25.0
    max_deadline_margin_ms: float = 80.0

    min_alternation_threshold_ms: float = 140.0
    max_alternation_threshold_ms: float = 240.0

    min_post_slider_gap_ms: float = 180.0
    max_post_slider_gap_ms: float = 320.0

    min_aim_lead_time_ms: float = 10.0
    max_aim_lead_time_ms: float = 26.0

    min_slider_step_ms: float = 16.0
    max_slider_step_ms: float = 32.0


class AdaptivePlanningPolicy:
    """
    Manages safe bounded adaptations to the planning engine hyperparameters.
    Monitors session outcomes to automatically rollback adjustments if accuracy regresses.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "lead_time_offset_ms": 0.0,
        "preparation_lead_ms": 8.0,
        "slider_waypoint_density": 16,
        "aim_smoothing_steps": 3,
        "deadline_margin_ms": 40.0,
        "alternation_threshold_ms": 200.0,
        "post_slider_gap_ms": 260.0,
        "aim_lead_time_ms": 16.0,
        "slider_step_ms": 22.0,
    }

    def __init__(self, bounds: Optional[PolicyBounds] = None):
        self.bounds = bounds or PolicyBounds()
        self.params: Dict[str, Any] = dict(self.DEFAULT_PARAMS)
        self._baseline_params: Dict[str, Any] = dict(self.DEFAULT_PARAMS)
        self._last_accuracy: float = 100.0
        self._history: List[Dict[str, Any]] = []
        self.load_policy()

    @property
    def policy_file(self) -> Path:
        root = Path(__file__).resolve().parents[4]
        if "PYTEST_CURRENT_TEST" in os.environ or os.environ.get("OSU_TEST_ENV") == "1":
            target = root / "data" / "osu_test_adaptive_policy.json"
        else:
            target = root / "data" / "osu_adaptive_policy.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def load_policy(self) -> Dict[str, Any]:
        p_file = self.policy_file
        if p_file.exists():
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                for k, v in saved.items():
                    if k in self.DEFAULT_PARAMS:
                        self.params[k] = v
                self._baseline_params = dict(self.params)
            except Exception:
                pass
        return dict(self.params)

    def save_policy(self) -> None:
        p_file = self.policy_file
        try:
            with open(p_file, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
        except Exception:
            pass

    def get_param(self, name: str, default: Any = None) -> Any:
        return self.params.get(name, default)

    def get_all_params(self) -> Dict[str, Any]:
        return dict(self.params)

    def update_from_diagnosis(
        self,
        diagnosis: DiagnosisReport,
        session_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Adjust planning parameters incrementally based on diagnostic findings.
        """
        # Save snapshot before modifying
        self._baseline_params = dict(self.params)

        cat = diagnosis.primary_category
        adjustments: Dict[str, Any] = {}

        if cat == ErrorCategory.TIMING_LATE:
            cur = self.params["lead_time_offset_ms"]
            new_val = min(self.bounds.max_lead_offset_ms, cur + 0.5)
            self.params["lead_time_offset_ms"] = round(new_val, 2)
            adjustments["lead_time_offset_ms"] = round(new_val, 2)

        elif cat == ErrorCategory.TIMING_EARLY:
            cur = self.params["lead_time_offset_ms"]
            new_val = max(self.bounds.min_lead_offset_ms, cur - 0.5)
            self.params["lead_time_offset_ms"] = round(new_val, 2)
            adjustments["lead_time_offset_ms"] = round(new_val, 2)

        elif cat == ErrorCategory.AIM_ERROR:
            cur_smooth = self.params["aim_smoothing_steps"]
            cur_prep = self.params["preparation_lead_ms"]
            cur_lead = self.params.get("aim_lead_time_ms", 16.0)
            self.params["aim_smoothing_steps"] = min(self.bounds.max_aim_smoothing_steps, cur_smooth + 1)
            self.params["preparation_lead_ms"] = min(self.bounds.max_prep_lead_ms, round(cur_prep + 1.0, 1))
            self.params["aim_lead_time_ms"] = min(self.bounds.max_aim_lead_time_ms, round(cur_lead + 2.0, 1))
            adjustments["aim_smoothing_steps"] = self.params["aim_smoothing_steps"]
            adjustments["preparation_lead_ms"] = self.params["preparation_lead_ms"]
            adjustments["aim_lead_time_ms"] = self.params["aim_lead_time_ms"]

        elif cat == ErrorCategory.SLIDER_TRACKING_ERROR:
            cur_density = self.params["slider_waypoint_density"]
            self.params["slider_waypoint_density"] = min(self.bounds.max_slider_density, cur_density + 2)
            adjustments["slider_waypoint_density"] = self.params["slider_waypoint_density"]

        elif cat in (ErrorCategory.ACTION_DEADLINE_MISS, ErrorCategory.QUEUE_OVERLOAD):
            cur_deadline = self.params["deadline_margin_ms"]
            cur_step = self.params.get("slider_step_ms", 22.0)
            self.params["deadline_margin_ms"] = min(self.bounds.max_deadline_margin_ms, round(cur_deadline + 5.0, 1))
            self.params["slider_step_ms"] = min(self.bounds.max_slider_step_ms, round(cur_step + 3.0, 1))
            adjustments["deadline_margin_ms"] = self.params["deadline_margin_ms"]
            adjustments["slider_step_ms"] = self.params["slider_step_ms"]

        elif cat == ErrorCategory.KEY_STATE_ERROR:
            cur_alt = self.params.get("alternation_threshold_ms", 200.0)
            cur_post = self.params.get("post_slider_gap_ms", 260.0)
            new_alt = min(self.bounds.max_alternation_threshold_ms, cur_alt + 15.0)
            new_post = min(self.bounds.max_post_slider_gap_ms, cur_post + 20.0)
            self.params["alternation_threshold_ms"] = round(new_alt, 1)
            self.params["post_slider_gap_ms"] = round(new_post, 1)
            adjustments["alternation_threshold_ms"] = self.params["alternation_threshold_ms"]
            adjustments["post_slider_gap_ms"] = self.params["post_slider_gap_ms"]

        self._history.append({
            "category": cat.value,
            "adjustments": adjustments,
            "params_after": dict(self.params),
        })

        if adjustments:
            self.save_policy()

        return adjustments

    def check_and_rollback_regression(
        self,
        current_accuracy: float,
        previous_accuracy: float,
        threshold_drop: float = 6.0,
    ) -> bool:
        """
        Detects if recent parameter adjustment caused a substantial performance regression.
        If accuracy drops by more than `threshold_drop`, roll back parameters to baseline.
        """
        if previous_accuracy - current_accuracy >= threshold_drop:
            # Regression confirmed -> restore baseline
            self.params = dict(self._baseline_params)
            self.save_policy()
            return True
        return False

    def reset(self) -> None:
        """Reset policy back to safe defaults."""
        self.params = dict(self.DEFAULT_PARAMS)
        self._baseline_params = dict(self.DEFAULT_PARAMS)
        self._history.clear()
        self.save_policy()


adaptive_policy = AdaptivePlanningPolicy()
