"""
Strategy Learner and Replay Evaluator for Shiori AI osu! Gameplay.
==================================================================
Performs contextual optimization based on beatmap archetypes and skill memory,
and supports offline replay evaluation of candidate parameters against recorded sessions.
"""

from typing import Any, Dict, List, Optional
import math


class StrategyLearner:
    """
    Synthesizes archetypal strategies and performs offline replay simulation
    to predict performance under different calibration parameters.
    """

    ARCHETYPE_STRATEGIES: Dict[str, Dict[str, Any]] = {
        "stream_speed": {
            "aim_smoothing_steps": 4,
            "preparation_lead_ms": 6.0,
            "deadline_margin_ms": 30.0,
            "notes": "Stream tốc độ cao: Tối ưu hóa phản xạ xen kẽ ngón và làm mượt quỹ đạo lia chuột.",
        },
        "jump_aim": {
            "aim_smoothing_steps": 2,
            "preparation_lead_ms": 10.0,
            "deadline_margin_ms": 45.0,
            "notes": "Jump Aim góc rộng: Tăng tốc độ chuẩn bị trước nốt (snap aim) để đón đầu vòng thu nhỏ.",
        },
        "rhythm_percussion": {
            "aim_smoothing_steps": 1,
            "preparation_lead_ms": 8.0,
            "deadline_margin_ms": 35.0,
            "notes": "Taiko bộ gõ: Khóa cứng độ trễ phần cứng và kích hoạt xen kẽ hai tay D/F/J/K.",
        },
        "catch_hyperdash": {
            "aim_smoothing_steps": 1,
            "preparation_lead_ms": 12.0,
            "deadline_margin_ms": 50.0,
            "notes": "Catch trái cây: Tính toán khoảng cách siêu tốc Hyperdash kẹp chặt giới hạn vận tốc đĩa hứng.",
        },
        "chord_alternation": {
            "aim_smoothing_steps": 1,
            "preparation_lead_ms": 8.0,
            "deadline_margin_ms": 30.0,
            "notes": "Mania hợp âm: Cấu hình đa làn động từ 1K đến 10K phím bấm đồng thời.",
        },
    }

    def recommend_strategy(
        self,
        archetype: str,
        skill_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a tailored parameter recommendation based on archetype and skill memory.
        """
        arch_key = str(archetype or "jump_aim").lower()
        base_strat = dict(self.ARCHETYPE_STRATEGIES.get(arch_key, self.ARCHETYPE_STRATEGIES["jump_aim"]))

        if skill_profile:
            prof_score = float(skill_profile.get("proficiency_score", 0.5))
            bias = float(skill_profile.get("timing_bias_ms", 0.0))

            # If proficiency is low, provide more generous deadline margins
            if prof_score < 0.6:
                base_strat["deadline_margin_ms"] = min(60.0, base_strat["deadline_margin_ms"] + 10.0)

            # Compensate for established skill timing bias
            base_strat["recommended_lead_offset_ms"] = round(-bias * 0.4, 2)
        else:
            base_strat["recommended_lead_offset_ms"] = 0.0

        return base_strat

    def evaluate_replay_session(
        self,
        session_data: Dict[str, Any],
        candidate_offset_ms: float,
    ) -> Dict[str, float]:
        """
        Offline replay evaluation: Estimates accuracy and Unstable Rate (UR)
        if candidate_offset_ms had been applied to recorded timing errors.
        """
        errors = list(session_data.get("timing_errors", []))
        if not errors:
            # Fallback estimation using average error
            avg_err = float(session_data.get("avg_timing_error_ms", 0.0))
            errors = [avg_err]

        # Shift all timing errors by candidate adjustment:
        # A positive candidate offset fires earlier, reducing positive error (late hit)
        adjusted_errors = [e - candidate_offset_ms for e in errors]

        n = len(adjusted_errors)
        mean_err = sum(adjusted_errors) / n
        variance = sum((x - mean_err) ** 2 for x in adjusted_errors) / max(1, n - 1)
        std_dev = math.sqrt(variance)
        estimated_ur = std_dev * 10.0

        # Estimate 300 / 100 / 50 counts based on standard OD 8 hit windows (300: +-31.5ms, 100: +-67.5ms)
        sim_300 = sum(1 for e in adjusted_errors if abs(e) <= 31.5)
        sim_100 = sum(1 for e in adjusted_errors if 31.5 < abs(e) <= 67.5)
        sim_50 = sum(1 for e in adjusted_errors if 67.5 < abs(e) <= 103.5)
        sim_miss = sum(1 for e in adjusted_errors if abs(e) > 103.5)

        total_sim = n
        if total_sim > 0:
            sim_acc = ((sim_300 * 300.0 + sim_100 * 100.0 + sim_50 * 50.0) / (total_sim * 300.0)) * 100.0
        else:
            sim_acc = 100.0

        return {
            "candidate_offset_ms": round(candidate_offset_ms, 2),
            "simulated_accuracy": round(sim_acc, 2),
            "simulated_unstable_rate": round(estimated_ur, 2),
            "simulated_mean_error_ms": round(mean_err, 2),
        }


strategy_learner = StrategyLearner()
