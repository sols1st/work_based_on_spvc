"""Analytic braking safety filter for the minimal AEBS experiment."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from stable_baselines3 import PPO

from Aebs.semantic.robust_controller import evaluate, load_contract


class SafetyFilteredController:
    """Raise the PPO braking action when conservative stopping distance requires it."""

    def __init__(
        self,
        baseline: PPO,
        distance_scale: float,
        perception_radius_m: float,
        radius_bin_edges_m: Optional[Sequence[float]] = None,
        radius_values_m: Optional[Sequence[float]] = None,
        safety_distance_m: float = 6.0,
        target_speed: float = 0.45,
        distance_buffer_m: float = 0.25,
        max_braking: float = 3.0,
        prevent_filter_overshoot: bool = False,
        dt: float = 0.05,
    ):
        self.baseline = baseline
        self.distance_scale = float(distance_scale)
        self.perception_radius_m = float(perception_radius_m)
        self.radius_bin_edges_m = (
            None
            if radius_bin_edges_m is None
            else np.asarray(radius_bin_edges_m, dtype=np.float32)
        )
        self.radius_values_m = (
            None if radius_values_m is None else np.asarray(radius_values_m, dtype=np.float32)
        )
        if (self.radius_bin_edges_m is None) != (self.radius_values_m is None):
            raise ValueError("radius_bin_edges_m and radius_values_m must be provided together")
        if self.radius_values_m is not None:
            if len(self.radius_bin_edges_m) != len(self.radius_values_m) + 1:
                raise ValueError("radius bin edges must have one more entry than radius values")
            if np.any(np.diff(self.radius_bin_edges_m) <= 0.0):
                raise ValueError("radius bin edges must be strictly increasing")
            if np.any(self.radius_values_m < 0.0):
                raise ValueError("perception radii must be non-negative")
        self.safety_distance_m = float(safety_distance_m)
        self.target_speed = float(target_speed)
        self.distance_buffer_m = float(distance_buffer_m)
        self.max_braking = float(max_braking)
        self.prevent_filter_overshoot = bool(prevent_filter_overshoot)
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @property
    def radius_mode(self) -> str:
        return "observed_distance_bin" if self.radius_values_m is not None else "global_max"

    def perception_radius(self, observed_distance_m: np.ndarray) -> np.ndarray:
        observed_distance_m = np.asarray(observed_distance_m, dtype=np.float32)
        if self.radius_values_m is None:
            return np.full_like(observed_distance_m, self.perception_radius_m)
        assignments = np.digitize(
            observed_distance_m,
            self.radius_bin_edges_m[1:-1],
            right=False,
        )
        assignments = np.clip(assignments, 0, len(self.radius_values_m) - 1)
        return self.radius_values_m[assignments]

    def predict_with_diagnostics(self, observations, deterministic: bool = True):
        values = np.asarray(observations, dtype=np.float32)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        baseline_actions, _ = self.baseline.predict(batch, deterministic=deterministic)
        baseline_actions = np.asarray(baseline_actions, dtype=np.float32).reshape(-1)

        observed_distance_m = batch[:, 0] * self.distance_scale
        perception_radius_m = self.perception_radius(observed_distance_m)
        speed = batch[:, 1]
        conservative_distance_m = observed_distance_m - perception_radius_m - self.distance_buffer_m
        braking_distance_m = np.maximum(conservative_distance_m - self.safety_distance_m, 1e-3)
        required_braking = np.maximum(speed**2 - self.target_speed**2, 0.0) / (2.0 * braking_distance_m)
        required_braking = np.clip(required_braking, 0.0, self.max_braking)
        uncapped_required_braking = required_braking.copy()
        uncapped_actions = np.maximum(baseline_actions, required_braking)
        actions = uncapped_actions.copy()
        if self.prevent_filter_overshoot:
            # The wrapped controller should approach the safe target speed,
            # not cross below it in one discrete step and strand the car.
            braking_to_target = np.maximum(speed - self.target_speed, 0.0) / self.dt
            actions = np.minimum(actions, braking_to_target)
        actions = actions.astype(np.float32).reshape(-1, 1)
        action_change = np.abs(actions.reshape(-1) - baseline_actions)
        extra_braking = np.maximum(actions.reshape(-1) - baseline_actions, 0.0)
        diagnostics = {
            "baseline_action": baseline_actions,
            "required_braking": required_braking,
            "uncapped_required_braking": uncapped_required_braking,
            "overshoot_cap_active": (
                uncapped_actions > actions.reshape(-1) + 1e-6
            ).astype(np.float32),
            "extra_braking": extra_braking,
            "intervened": (action_change > 1e-6).astype(np.float32),
            "perception_radius_m": perception_radius_m,
            "conservative_distance_m": conservative_distance_m,
        }
        if single:
            actions = actions[0]
            diagnostics = {key: value[0] for key, value in diagnostics.items()}
        return actions, diagnostics

    def predict(self, observations, deterministic: bool = True):
        actions, _ = self.predict_with_diagnostics(observations, deterministic=deterministic)
        return actions, None


def load_controller(path: str):
    """Load either a normal PPO zip or a JSON safety-filter specification."""
    if not path.endswith(".json"):
        return PPO.load(path, device="cpu")
    with open(path, "r", encoding="utf-8") as stream:
        spec = json.load(stream)
    if spec.get("type") != "analytic_aebs_safety_filter":
        raise ValueError(f"unsupported controller specification: {spec.get('type')}")
    return SafetyFilteredController(
        baseline=PPO.load(spec["baseline_controller"], device="cpu"),
        distance_scale=spec["distance_scale"],
        perception_radius_m=spec["perception_radius_m"],
        radius_bin_edges_m=spec.get("radius_bin_edges_m"),
        radius_values_m=spec.get("radius_values_m"),
        safety_distance_m=spec["safety_distance_m"],
        target_speed=spec["target_speed"],
        distance_buffer_m=spec["distance_buffer_m"],
        max_braking=spec["max_braking"],
        prevent_filter_overshoot=spec.get("prevent_filter_overshoot", False),
        dt=spec.get("dt", 0.05),
    )


def build_and_evaluate(
    baseline_controller: str,
    semantic_checkpoint: str,
    output_dir: Path,
    episodes: int,
    seed: int,
    target_speed: float,
    distance_buffer_m: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    perception_radius_m = float(np.max(np.asarray(contract.radii)) * distance_scale)
    spec = {
        "type": "analytic_aebs_safety_filter",
        "baseline_controller": baseline_controller,
        "distance_scale": distance_scale,
        "perception_radius_m": perception_radius_m,
        "safety_distance_m": 6.0,
        "target_speed": float(target_speed),
        "distance_buffer_m": float(distance_buffer_m),
        "max_braking": 3.0,
    }
    spec_path = output_dir / "safety_filter.json"
    with open(spec_path, "w", encoding="utf-8") as stream:
        json.dump(spec, stream, indent=2, ensure_ascii=False)
    controller = load_controller(str(spec_path))
    modes = ("exact", "uniform", "random_boundary", "worst_endpoint")
    metrics = {
        "experiment": "analytic_safety_filter_mvp",
        "seed": int(seed),
        "episodes": int(episodes),
        "spec": spec,
        "baseline": {},
        "filtered": {},
    }
    baseline = PPO.load(baseline_controller, device="cpu")
    for mode in modes:
        metrics["baseline"][mode] = evaluate(baseline, contract, distance_scale, mode, episodes, seed)
        metrics["filtered"][mode] = evaluate(controller, contract, distance_scale, mode, episodes, seed)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-controller", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--output-dir", default="results/mvp/02_safety_filter")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-speed", type=float, default=0.45)
    parser.add_argument("--distance-buffer-m", type=float, default=0.25)
    args = parser.parse_args()
    metrics = build_and_evaluate(
        args.baseline_controller,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.episodes,
        args.seed,
        args.target_speed,
        args.distance_buffer_m,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
