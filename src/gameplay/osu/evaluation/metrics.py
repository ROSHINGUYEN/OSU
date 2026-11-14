"""
Gameplay Metrics and Accuracy Calculator for Shiori AI osu!.
Tracks hit distributions, accuracy percentage, combo, and Unstable Rate (UR).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GameplayMetrics:
    """Session performance statistics and accuracy metrics."""
    count_300: int = 0
    count_100: int = 0
    count_50: int = 0
    count_miss: int = 0
    count_max: int = 0  # Mania MAX
    current_combo: int = 0
    max_combo: int = 0
    total_score: int = 0
    timing_errors_ms: List[float] = field(default_factory=list)

    @property
    def total_hits(self) -> int:
        return self.count_300 + self.count_100 + self.count_50 + self.count_miss + self.count_max

    @property
    def hits(self) -> Dict[str, int]:
        return {
            "300": self.count_300,
            "100": self.count_100,
            "50": self.count_50,
            "miss": self.count_miss,
            "max": self.count_max,
        }

    _custom_accuracy: Optional[float] = None

    @property
    def accuracy(self) -> float:
        """Standard osu! accuracy formula or synchronized in-game accuracy."""
        if self._custom_accuracy is not None:
            return self._custom_accuracy
        total = self.total_hits
        if total == 0:
            return 100.0

        # Weights: 300=1.0, 100=1/3, 50=1/6, miss=0
        points = (
            (self.count_300 + self.count_max) * 300.0 +
            self.count_100 * 100.0 +
            self.count_50 * 50.0
        )
        max_points = total * 300.0
        return (points / max_points) * 100.0

    @accuracy.setter
    def accuracy(self, value: float) -> None:
        self._custom_accuracy = float(value)

    @property
    def timing_error_mean(self) -> float:
        """Mean timing error in milliseconds (signed: positive=late, negative=early)."""
        if not self.timing_errors_ms:
            return 0.0
        return sum(self.timing_errors_ms) / len(self.timing_errors_ms)

    @property
    def timing_error_variance(self) -> float:
        """Sample variance of timing errors in milliseconds squared."""
        if len(self.timing_errors_ms) < 2:
            return 0.0
        n = len(self.timing_errors_ms)
        mean = sum(self.timing_errors_ms) / n
        return sum((x - mean) ** 2 for x in self.timing_errors_ms) / (n - 1)

    @property
    def unstable_rate(self) -> float:
        """
        Unstable Rate (UR): Standard deviation of timing errors in milliseconds * 10.
        Lower UR means tighter and more consistent rhythm precision.
        """
        var = self.timing_error_variance
        if var <= 0.0:
            return 0.0
        std_dev = math.sqrt(var)
        return std_dev * 10.0

    def record_result(self, result: str, delta_ms: float = 0.0) -> None:
        """Record a single hit or miss result."""
        result = result.upper()
        if result in ["300", "MAX"]:
            if result == "MAX":
                self.count_max += 1
            else:
                self.count_300 += 1
            self.current_combo += 1
            self.total_score += 300 * max(1, self.current_combo)
        elif result == "100":
            self.count_100 += 1
            self.current_combo += 1
            self.total_score += 100 * max(1, self.current_combo)
        elif result == "50":
            self.count_50 += 1
            self.current_combo += 1
            self.total_score += 50 * max(1, self.current_combo)
        else:  # Miss
            self.count_miss += 1
            self.current_combo = 0

        self.max_combo = max(self.max_combo, self.current_combo)

        if result != "MISS":
            self.timing_errors_ms.append(delta_ms)
            if len(self.timing_errors_ms) > 500:
                self.timing_errors_ms.pop(0)

    def reset(self) -> None:
        self.count_300 = 0
        self.count_100 = 0
        self.count_50 = 0
        self.count_miss = 0
        self.count_max = 0
        self.current_combo = 0
        self.max_combo = 0
        self.total_score = 0
        self.timing_errors_ms.clear()
        self._custom_accuracy = None
