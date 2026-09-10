"""A small 4x4 state-dependent disturbance model for the AEBS MVP."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class StateDependentUncertainty:
    distance_edges_m: np.ndarray
    speed_edges: np.ndarray
    mean: np.ndarray
    support_low: np.ndarray
    support_high: np.ndarray
    mean_radius: np.ndarray
    counts: np.ndarray

    def cell_index(self, distance_m: np.ndarray, speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_index = np.digitize(distance_m, self.distance_edges_m[1:-1], right=False)
        v_index = np.digitize(speed, self.speed_edges[1:-1], right=False)
        d_index = np.clip(d_index, 0, len(self.distance_edges_m) - 2)
        v_index = np.clip(v_index, 0, len(self.speed_edges) - 2)
        return d_index, v_index

    def lookup(self, state: np.ndarray, distance_scale: float) -> Dict[str, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        distance_m = state[..., 0] * distance_scale
        speed = state[..., 1]
        d_index, v_index = self.cell_index(distance_m, speed)
        return {
            "mean": self.mean[d_index, v_index],
            "support_low": self.support_low[d_index, v_index],
            "support_high": self.support_high[d_index, v_index],
            "mean_radius": self.mean_radius[d_index, v_index],
            "counts": self.counts[d_index, v_index],
        }

    def to_npz(self, path: str) -> None:
        np.savez(
            path,
            distance_edges_m=self.distance_edges_m,
            speed_edges=self.speed_edges,
            mean=self.mean,
            support_low=self.support_low,
            support_high=self.support_high,
            mean_radius=self.mean_radius,
            counts=self.counts,
        )

    @classmethod
    def from_npz(cls, path: str) -> "StateDependentUncertainty":
        data = np.load(path)
        return cls(
            distance_edges_m=data["distance_edges_m"],
            speed_edges=data["speed_edges"],
            mean=data["mean"],
            support_low=data["support_low"],
            support_high=data["support_high"],
            mean_radius=data["mean_radius"],
            counts=data["counts"],
        )

