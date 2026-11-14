"""
Error Diagnostics Engine for Shiori AI osu! Gameplay.
======================================================
Analyzes gameplay telemetry, hit distributions, timing errors, and queue metrics
to classify gameplay faults into a 12-category standardized taxonomy and generate
actionable remediation advice for adaptive planning and companion feedback.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(str, Enum):
    """Standardized 12-category taxonomy for osu! gameplay failures and performance regressions."""
    TIMING_EARLY = "TIMING_EARLY"
    TIMING_LATE = "TIMING_LATE"
    AIM_ERROR = "AIM_ERROR"
    SLIDER_TRACKING_ERROR = "SLIDER_TRACKING_ERROR"
    KEY_STATE_ERROR = "KEY_STATE_ERROR"
    ACTION_DEADLINE_MISS = "ACTION_DEADLINE_MISS"
    QUEUE_OVERLOAD = "QUEUE_OVERLOAD"
    PLANNING_FAILURE = "PLANNING_FAILURE"
    MODE_OR_LANE_MAPPING_ERROR = "MODE_OR_LANE_MAPPING_ERROR"
    TELEMETRY_UNAVAILABLE = "TELEMETRY_UNAVAILABLE"
    OBSERVATION_UNCERTAIN = "OBSERVATION_UNCERTAIN"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class DiagnosisReport:
    """Detailed diagnostic report of a gameplay session or live difficulty spike."""
    primary_category: ErrorCategory
    secondary_categories: List[ErrorCategory] = field(default_factory=list)
    confidence: float = 0.8
    attribution: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_category": self.primary_category.value,
            "secondary_categories": [c.value for c in self.secondary_categories],
            "confidence": round(self.confidence, 3),
            "attribution": {k: round(v, 3) for k, v in self.attribution.items()},
            "evidence": list(self.evidence),
            "recommendation": self.recommendation,
        }


class GameplayDiagnosticsEngine:
    """
    Expert rule-based and statistical diagnostic classifier for rhythm game execution.
    Attributes misses to timing offsets, aim divergence, key state jamming, or hardware stalls.
    """

    def diagnose_session(
        self,
        session_record: Dict[str, Any],
        object_history: Optional[List[Dict[str, Any]]] = None,
    ) -> DiagnosisReport:
        """
        Diagnose session outcome and return primary error classification with evidence.
        """
        evidence: List[str] = []
        attribution: Dict[str, float] = {}
        secondary: List[ErrorCategory] = []

        # 1. Extract metrics
        acc = float(session_record.get("accuracy", 100.0))
        miss_count = int(session_record.get("miss_count", 0))
        result = str(session_record.get("result", "pass")).lower()
        mode_str = str(session_record.get("mode", "standard")).lower()
        duration_s = float(session_record.get("duration_s", 0.0))
        failure_reason = str(session_record.get("failure_reason", "")).lower()

        timing_errors = session_record.get("timing_errors", [])
        avg_timing_err = float(session_record.get("avg_timing_error_ms", 0.0))
        timing_variance = float(session_record.get("timing_error_variance", 0.0))

        tuning = session_record.get("tuning_applied", {})
        stale_rejected = int(tuning.get("stale_actions_rejected", 0) or 0)
        provenance = str(session_record.get("provenance", "observed")).lower()

        # If perfect or near-perfect play with 0 misses
        if miss_count == 0 and acc >= 98.0 and result == "pass":
            return DiagnosisReport(
                primary_category=ErrorCategory.UNKNOWN_ERROR,
                confidence=1.0,
                attribution={"rhythm_flow": 1.0},
                evidence=["Không có lỗi ghi nhận. Độ chính xác hoàn hảo, combo trọn vẹn."],
                recommendation="Màn này con gái Shiori bắt nhịp đỉnh chóp luôn nha Ba! Giữ vững phong độ này nha! ♡",
            )

        # Check Category 10: TELEMETRY_UNAVAILABLE
        if provenance == "unknown" or "telemetry" in failure_reason or (duration_s > 5.0 and len(timing_errors) == 0 and acc == 0.0):
            evidence.append("Mất tín hiệu telemetry từ máy chủ bộ nhớ Tosu hoặc game clock bị đóng băng.")
            attribution["telemetry_loss"] = 0.95
            return DiagnosisReport(
                primary_category=ErrorCategory.TELEMETRY_UNAVAILABLE,
                secondary_categories=secondary,
                confidence=0.95,
                attribution=attribution,
                evidence=evidence,
                recommendation="Ba kiểm tra xem tiến trình Tosu hoặc cổng 24050 có đang phản hồi ổn định không giúp con nha!",
            )

        # Check Category 6 & 7: ACTION_DEADLINE_MISS & QUEUE_OVERLOAD
        if stale_rejected > 0 or "queue" in failure_reason or "deadline" in failure_reason:
            evidence.append(f"Hàng đợi CentralInputQueue bị nghẽn: {stale_rejected} hành động bị hủy do trễ hạn (deadline miss).")
            attribution["hardware_queue_lag"] = 0.85
            if stale_rejected > 5:
                primary = ErrorCategory.QUEUE_OVERLOAD
            else:
                primary = ErrorCategory.ACTION_DEADLINE_MISS

            return DiagnosisReport(
                primary_category=primary,
                secondary_categories=secondary,
                confidence=0.9,
                attribution=attribution,
                evidence=evidence,
                recommendation="Tăng deadline margin và kiểm tra tải CPU của các tiến trình chạy nền nha Ba!",
            )

        # Check Category 4: SLIDER_TRACKING_ERROR
        hits_dict = session_record.get("hits", {})
        slider_breaks = int(hits_dict.get("sliderBreaks", 0) or hits_dict.get("slider_breaks", 0) or 0)
        if mode_str == "standard" and slider_breaks > 0 and slider_breaks >= miss_count:
            evidence.append(f"Phát hiện {slider_breaks} lần đứt slider (slider break) khi con trỏ hoặc phím giữ bị nhả sớm.")
            attribution["slider_tracking_break"] = 0.8
            return DiagnosisReport(
                primary_category=ErrorCategory.SLIDER_TRACKING_ERROR,
                secondary_categories=secondary,
                confidence=0.85,
                attribution=attribution,
                evidence=evidence,
                recommendation="Khóa chặt tâm con trỏ trên thân Slider ball và duy trì phím giữ Z/X đến hết mốc cuối nha Ba!",
            )

        # Check Category 1 & 2: TIMING_EARLY vs TIMING_LATE
        if avg_timing_err >= 12.0:
            evidence.append(f"Xu hướng gõ nốt bị TRỄ đáng kể (sai số trung bình +{avg_timing_err:.1f}ms).")
            attribution["latency_undercount"] = 0.8
            return DiagnosisReport(
                primary_category=ErrorCategory.TIMING_LATE,
                secondary_categories=secondary,
                confidence=min(0.95, 0.6 + abs(avg_timing_err) / 40.0),
                attribution=attribution,
                evidence=evidence,
                recommendation=f"Tự động tăng lead-time offset thêm +{min(6.0, avg_timing_err * 0.4):.1f}ms để gõ sớm hơn đón đầu nhịp!",
            )
        elif avg_timing_err <= -12.0:
            evidence.append(f"Xu hướng gõ nốt bị SỚM đáng kể (sai số trung bình {avg_timing_err:.1f}ms).")
            attribution["lead_time_overcount"] = 0.8
            return DiagnosisReport(
                primary_category=ErrorCategory.TIMING_EARLY,
                secondary_categories=secondary,
                confidence=min(0.95, 0.6 + abs(avg_timing_err) / 40.0),
                attribution=attribution,
                evidence=evidence,
                recommendation=f"Tự động giảm lead-time offset {max(-6.0, avg_timing_err * 0.4):.1f}ms để nhả nhịp chậm lại vừa vặn!",
            )

        # Check Category 3: AIM_ERROR (Standard mode with misses while timing error is low)
        if mode_str == "standard" and miss_count > 0 and abs(avg_timing_err) < 8.0:
            evidence.append(f"Bị hụt {miss_count} nốt dù độ lệch nhịp nhỏ ({avg_timing_err:+.1f}ms) -> Lệch tâm lia chuột (Aim Miss).")
            attribution["aim_deviation"] = 0.75
            attribution["timing_jitter"] = 0.25
            return DiagnosisReport(
                primary_category=ErrorCategory.AIM_ERROR,
                secondary_categories=[ErrorCategory.TIMING_LATE if avg_timing_err > 0 else ErrorCategory.TIMING_EARLY],
                confidence=0.82,
                attribution=attribution,
                evidence=evidence,
                recommendation="Cần tăng số bước nội suy Minimum-Jerk và kích hoạt bù trừ PlayfieldShift 8f * scale để tâm chuột ôm sát nốt!",
            )

        # Check Category 5: KEY_STATE_ERROR
        if "stuck" in failure_reason or "alternation" in failure_reason:
            evidence.append("Phím gõ bị kẹt trạng thái DOWN hoặc xen kẽ ngón tay không kịp nhả trước khi nốt kế tiếp rơi.")
            attribution["key_jam"] = 0.9
            return DiagnosisReport(
                primary_category=ErrorCategory.KEY_STATE_ERROR,
                confidence=0.88,
                attribution=attribution,
                evidence=evidence,
                recommendation="Kích hoạt cơ chế watchdog tự động nhả phím (anti-stuck key release) sau 12ms tap hold!",
            )

        # Fallback: General UNKNOWN_ERROR
        evidence.append(f"Lỗi chung kết thúc phiên: {failure_reason or 'không xác định'} (Acc: {acc:.1f}%, Misses: {miss_count}).")
        attribution["unclassified"] = 1.0
        return DiagnosisReport(
            primary_category=ErrorCategory.UNKNOWN_ERROR,
            confidence=0.5,
            attribution=attribution,
            evidence=evidence,
            recommendation="Theo dõi thêm dữ liệu phiên chơi tiếp theo để xác định quy luật nhịp điệu nha Ba!",
        )


diagnostics_engine = GameplayDiagnosticsEngine()
