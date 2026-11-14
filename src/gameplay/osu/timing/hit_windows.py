"""
Hit Window Models and Accuracy Calculations for all osu! Game Modes.
Accurately maps Overall Difficulty (OD) to millisecond tolerances.
"""

from typing import Optional, Tuple
from src.gameplay.osu.core.state_store import GameMode


class HitWindowModel:
    """Calculates hit window boundaries based on mode and Overall Difficulty (OD)."""

    @staticmethod
    def standard(od: float) -> Tuple[float, float, float]:
        """Returns (window_300, window_100, window_50) in milliseconds."""
        od = max(0.0, min(10.0, od))
        w300 = 80.0 - (6.0 * od)
        w100 = 140.0 - (8.0 * od)
        w50 = 200.0 - (10.0 * od)
        return w300, w100, w50

    @staticmethod
    def calculate_preempt_ms(ar: float) -> float:
        """
        Calculates Approach Rate preempt time in milliseconds.
        Matches ppy/osu OsuHitObject.cs:
          PREEMPT_MIN = 450 (AR 10)
          PREEMPT_MID = 1200 (AR 5)
          PREEMPT_MAX = 1800 (AR 0)
        """
        ar = max(0.0, min(11.0, ar))
        if ar < 5.0:
            return 1200.0 + 600.0 * (5.0 - ar) / 5.0
        elif ar == 5.0:
            return 1200.0
        else:
            return 1200.0 - 750.0 * (ar - 5.0) / 5.0

    @classmethod
    def approach_scale_to_time_ms(cls, scale: float, ar: float = 9.0) -> float:
        """
        Calculates milliseconds until hit from optical approach circle scale.
        In ppy/osu, the approach circle spawns at scale 4.0 at (t_hit - Preempt)
        and collapses linearly to scale 1.0 at t_hit.
        """
        preempt = cls.calculate_preempt_ms(ar)
        return max(0.0, (scale - 1.0) / 3.0 * preempt)

    @staticmethod
    def calculate_circle_radius(cs: float) -> float:
        """
        Calculates hit circle radius in osu! pixels from Circle Size (CS).
        Matches ppy/osu LegacyRulesetExtensions.CalculateScaleFromCircleSize:
          Radius = OBJECT_RADIUS * (0.85 - 0.07 * CS) = 54.4 - 4.48 * CS
        """
        cs = max(0.0, min(10.0, cs))
        return 54.4 - (4.48 * cs)

    @staticmethod
    def taiko(od: float) -> Tuple[float, float]:
        """
        Returns (window_great, window_good) in milliseconds.
        Matches ppy/osu TaikoHitWindows.cs exactly:
          GREAT_WINDOW_RANGE = DifficultyRange(50, 35, 20) → ±(floor(range)-0.5)
          OK_WINDOW_RANGE    = DifficultyRange(120, 80, 50) → ±(floor(range)-0.5)
        """
        od = max(0.0, min(10.0, od))

        def _difficulty_range(od: float, min_v: float, mid_v: float, max_v: float) -> float:
            if od > 5:
                return mid_v + (max_v - mid_v) * (od - 5) / 5
            if od < 5:
                return mid_v - (mid_v - min_v) * (5 - od) / 5
            return mid_v

        w_great = int(_difficulty_range(od, 50, 35, 20)) - 0.5
        w_good  = int(_difficulty_range(od, 120, 80, 50)) - 0.5
        return w_great, w_good

    @staticmethod
    def mania(od: float) -> Tuple[float, float, float, float, float]:
        """Returns (window_max, window_300, window_200, window_100, window_50) in milliseconds."""
        od = max(0.0, min(10.0, od))
        w_max = 16.0
        w_300 = 64.0 - (3.0 * od)
        w_200 = 97.0 - (3.0 * od)
        w_100 = 127.0 - (3.0 * od)
        w_50 = 151.0 - (3.0 * od)
        return w_max, w_300, w_200, w_100, w_50

    @classmethod
    def evaluate_timing_error(cls, delta_ms: float, od: float, mode: Optional[GameMode] = None) -> str:
        """
        Evaluates a hit delta in milliseconds for osu! Standard:
        delta_ms = hit_time_ms - target_time_ms.
        Returns score classification: '300', '100', '50', 'miss'.
        """
        abs_delta = abs(delta_ms)
        w300, w100, w50 = cls.standard(od)
        if abs_delta <= w300:
            return "300"
        elif abs_delta <= w100:
            return "100"
        elif abs_delta <= w50:
            return "50"
        return "miss"
