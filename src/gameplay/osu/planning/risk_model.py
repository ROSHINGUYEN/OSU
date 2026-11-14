"""
Risk Assessment Model for Early Pruning of Unsalvageable Notes.
Prevents catastrophic cascading misses by gracefully dropping unreachable objects.
"""

import math
from typing import Optional


class RiskModel:
    """Evaluates physical execution feasibility and subsequent note risk."""

    MAX_FEASIBLE_CURSOR_SPEED = 4.5  # Screen pixels per millisecond

    @classmethod
    def evaluate_reachability(
        cls,
        curr_x: float,
        curr_y: float,
        target_x: float,
        target_y: float,
        time_to_target_ms: float,
    ) -> float:
        """
        Returns estimated probability of reaching target in time: P(hit) in [0.0, 1.0].
        """
        if time_to_target_ms <= 0:
            return 0.0

        dist = math.hypot(target_x - curr_x, target_y - curr_y)
        req_speed = dist / time_to_target_ms

        if req_speed <= 1.5:
            return 0.99
        elif req_speed <= 3.0:
            return 0.90
        elif req_speed <= cls.MAX_FEASIBLE_CURSOR_SPEED:
            return 0.65
        else:
            # Over speed limit
            overshoot = req_speed - cls.MAX_FEASIBLE_CURSOR_SPEED
            return max(0.05, 0.5 - (overshoot * 0.2))

    @classmethod
    def is_unsalvageable(
        cls,
        curr_x: float,
        curr_y: float,
        target_x: float,
        target_y: float,
        time_to_target_ms: float,
        dist_to_next: float = 0.0,
        time_to_next_ms: float = 200.0,
    ) -> bool:
        """
        Determines if current note should be dropped to save the subsequent combo.
        """
        p_hit = cls.evaluate_reachability(curr_x, curr_y, target_x, target_y, time_to_target_ms)
        if p_hit > 0.25:
            return False  # Still salvageable

        # Check risk to next note if we attempt current note
        if time_to_next_ms > 0:
            speed_to_next = dist_to_next / time_to_next_ms
            if speed_to_next > 3.5:
                # High risk of cascading double-miss
                return True

        return False
