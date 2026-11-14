"""
Evaluation and Telemetry Domain Package for Shiori AI osu!.
Exports GameplayMetrics, FeedbackEvaluator, TelemetryLogger, OsuGameMemory,
GameplayDiagnosticsEngine, and StrategyLearner.
"""

from src.gameplay.osu.evaluation.diagnostics import (
    DiagnosisReport,
    ErrorCategory,
    GameplayDiagnosticsEngine,
    diagnostics_engine,
)
from src.gameplay.osu.evaluation.feedback import (
    FeedbackEvaluator,
    ObservationSource,
    feedback_evaluator,
)
from src.gameplay.osu.evaluation.game_memory import (
    OsuGameMemory,
    osu_game_memory,
)
from src.gameplay.osu.evaluation.metrics import GameplayMetrics
from src.gameplay.osu.evaluation.strategy_learner import (
    StrategyLearner,
    strategy_learner,
)
from src.gameplay.osu.evaluation.telemetry import (
    TelemetryLogger,
    telemetry_logger,
)

__all__ = [
    "GameplayMetrics",
    "FeedbackEvaluator",
    "feedback_evaluator",
    "ObservationSource",
    "TelemetryLogger",
    "telemetry_logger",
    "OsuGameMemory",
    "osu_game_memory",
    "ErrorCategory",
    "DiagnosisReport",
    "GameplayDiagnosticsEngine",
    "diagnostics_engine",
    "StrategyLearner",
    "strategy_learner",
]
