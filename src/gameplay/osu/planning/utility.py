"""
Risk-Aware Multi-Objective Utility Model for Action Candidate Selection.
Balances Accuracy, Combo Protection, Execution Cost, and Subsequent Note Risk.
"""

from typing import Dict
from src.gameplay.osu.input.action_model import ActionIntent


class UtilityEvaluator:
    """Evaluates utility score of action candidates."""

    def __init__(
        self,
        w_hit: float = 1.0,
        w_combo: float = 1.5,
        w_acc: float = 0.8,
        w_risk: float = 2.0,
        w_cost: float = 0.5,
    ):
        self.w_hit = w_hit
        self.w_combo = w_combo
        self.w_acc = w_acc
        self.w_risk = w_risk
        self.w_cost = w_cost

    def evaluate(
        self,
        action: ActionIntent,
        current_combo: int = 0,
        prob_hit: float = 0.95,
        expected_score: float = 300.0,
        risk_next: float = 0.1,
        transition_cost: float = 0.05,
    ) -> float:
        """
        Calculates utility:
        U = w_hit * P(hit) + w_combo * I_combo + w_acc * (Score/300) - w_risk * Risk - w_cost * Cost
        """
        # Combo protection multiplier: higher importance if holding large combo
        i_combo = 1.0 if current_combo > 50 else (current_combo / 50.0)
        norm_score = max(0.0, min(1.0, expected_score / 300.0))

        score = (
            (self.w_hit * prob_hit) +
            (self.w_combo * i_combo) +
            (self.w_acc * norm_score) -
            (self.w_risk * risk_next) -
            (self.w_cost * transition_cost)
        )
        return score
