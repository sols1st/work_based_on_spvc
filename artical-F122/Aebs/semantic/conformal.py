import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SplitConformalRegressor:
    """A finite-sample split-conformal interval for scalar regression.

    The nonconformity score is |y - mean| / scale.  Passing unit scales
    recovers an ordinary constant-width conformal interval.
    """

    alpha: float
    quantile: float
    calibration_size: int
    min_scale: float = 1e-4

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        alpha: float = 0.1,
        min_scale: float = 1e-4,
    ) -> "SplitConformalRegressor":
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if not (len(y_true) == len(mean) == len(scale)) or len(y_true) == 0:
            raise ValueError("calibration arrays must have the same non-zero length")
        scores = np.abs(y_true - mean) / np.maximum(scale, min_scale)
        # Exact split-conformal order statistic: ceil((n+1)(1-alpha)).
        rank = min(math.ceil((len(scores) + 1) * (1.0 - alpha)), len(scores))
        quantile = float(np.partition(scores, rank - 1)[rank - 1])
        return cls(alpha=alpha, quantile=quantile, calibration_size=len(scores), min_scale=min_scale)

    def interval(self, mean: np.ndarray, scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean)
        scale = np.maximum(np.asarray(scale), self.min_scale)
        radius = self.quantile * scale
        return mean - radius, mean + radius

    def coverage(self, y_true: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
        lower, upper = self.interval(mean, scale)
        y_true = np.asarray(y_true)
        return float(np.mean((y_true >= lower) & (y_true <= upper)))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class MondrianSplitConformalRegressor:
    """Split conformal with groups determined only by the point prediction."""

    alpha: float
    bin_edges: Tuple[float, ...]
    quantiles: Tuple[float, ...]
    calibration_sizes: Tuple[int, ...]
    min_scale: float = 1e-4

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
        alpha: float = 0.1,
        n_bins: int = 4,
        min_scale: float = 1e-4,
    ) -> "MondrianSplitConformalRegressor":
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
        scale = np.asarray(scale, dtype=np.float64).reshape(-1)
        if not (len(y_true) == len(mean) == len(scale)) or len(y_true) == 0:
            raise ValueError("calibration arrays must have the same non-zero length")
        edges = np.quantile(mean, np.linspace(0.0, 1.0, n_bins + 1))
        # Only interior edges affect assignment; infinities make deployment behavior explicit.
        edges[0], edges[-1] = -np.inf, np.inf
        assignments = np.digitize(mean, edges[1:-1], right=False)
        quantiles: List[float] = []
        sizes: List[int] = []
        for bin_index in range(n_bins):
            mask = assignments == bin_index
            if not np.any(mask):
                raise ValueError("empty Mondrian calibration bin")
            local = SplitConformalRegressor.fit(
                y_true[mask], mean[mask], scale[mask], alpha=alpha, min_scale=min_scale
            )
            quantiles.append(local.quantile)
            sizes.append(local.calibration_size)
        return cls(
            alpha=alpha,
            bin_edges=tuple(float(value) for value in edges),
            quantiles=tuple(quantiles),
            calibration_sizes=tuple(sizes),
            min_scale=min_scale,
        )

    def interval(self, mean: np.ndarray, scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean)
        scale = np.maximum(np.asarray(scale), self.min_scale)
        assignments = np.digitize(mean, np.asarray(self.bin_edges[1:-1]), right=False)
        quantile = np.asarray(self.quantiles)[assignments]
        radius = quantile * scale
        return mean - radius, mean + radius

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["bin_edges"] = list(self.bin_edges)
        result["quantiles"] = list(self.quantiles)
        result["calibration_sizes"] = list(self.calibration_sizes)
        return result


@dataclass(frozen=True)
class StateConditionalErrorContract:
    """Piecewise-constant bound on perception error as a function of true state."""

    alpha: float
    bin_edges: Tuple[float, ...]
    radii: Tuple[float, ...]
    calibration_sizes: Tuple[int, ...]

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        estimate: np.ndarray,
        state: np.ndarray,
        bin_edges: np.ndarray,
        alpha: float,
    ) -> "StateConditionalErrorContract":
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        edges = np.asarray(bin_edges, dtype=np.float64).reshape(-1)
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin_edges must be strictly increasing")
        assignments = np.digitize(state, edges[1:-1], right=False)
        radii, sizes = [], []
        for bin_index in range(len(edges) - 1):
            mask = assignments == bin_index
            if not np.any(mask):
                raise ValueError("empty state-conditional calibration bin")
            local = SplitConformalRegressor.fit(
                y_true[mask], estimate[mask], np.ones(mask.sum()), alpha=alpha
            )
            radii.append(local.quantile)
            sizes.append(local.calibration_size)
        return cls(
            alpha=alpha,
            bin_edges=tuple(float(value) for value in edges),
            radii=tuple(radii),
            calibration_sizes=tuple(sizes),
        )

    def radius(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state)
        assignments = np.digitize(state, np.asarray(self.bin_edges[1:-1]), right=False)
        assignments = np.clip(assignments, 0, len(self.radii) - 1)
        return np.asarray(self.radii)[assignments]

    def estimate_interval(
        self, true_semantic: np.ndarray, conditioning_state: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if conditioning_state is None:
            conditioning_state = true_semantic
        radius = self.radius(conditioning_state)
        return np.asarray(true_semantic) - radius, np.asarray(true_semantic) + radius

    def to_dict(self) -> Dict[str, object]:
        result = asdict(self)
        result["bin_edges"] = list(self.bin_edges)
        result["radii"] = list(self.radii)
        result["calibration_sizes"] = list(self.calibration_sizes)
        return result
