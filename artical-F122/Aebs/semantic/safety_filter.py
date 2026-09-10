"""Analytic braking safety filter for the minimal AEBS experiment."""

import argparse
import json
from pathlib import Path
from typing import Dict

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
        safety_distance_m: float = 6.0,
        target_speed: float = 0.45,
        distance_buffer_m: float = 0.25,
        max_braking: float = 3.0,
    ):
        self.baseline = baseline
        self.distance_scale = float(distance_scale)
        self.perception_radius_m = float(perception_radius_m)
        self.safety_distance_m = float(safety_distance_m)
        self.target_speed = float(target_speed)
        self.distance_buffer_m = float(distance_buffer_m)
        self.max_braking = float(max_braking)

    def predict(self, observations, deterministic: bool = True):
        values = np.asarray(observations, dtype=np.float32)
        single = values.ndim == 1
        batch = values.reshape(1, -1) if single else values
        baseline_actions, _ = self.baseline.predict(batch, deterministic=deterministic)
        baseline_actions = np.asarray(baseline_actions, dtype=np.float32).reshape(-1)

        observed_distance_m = batch[:, 0] * self.distance_scale
        speed = batch[:, 1]
        conservative_distance_m = observed_distance_m - self.perception_radius_m - self.distance_buffer_m
        braking_distance_m = np.maximum(conservative_distance_m - self.safety_distance_m, 1e-3)
        required_braking = np.maximum(speed**2 - self.target_speed**2, 0.0) / (2.0 * braking_distance_m)
        required_braking = np.clip(required_braking, 0.0, self.max_braking)
        actions = np.maximum(baseline_actions, required_braking).astype(np.float32).reshape(-1, 1)
        if single:
            actions = actions[0]
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
        PPO.load(spec["baseline_controller"], device="cpu"),
        spec["distance_scale"],
        spec["perception_radius_m"],
        spec["safety_distance_m"],
        spec["target_speed"],
        spec["distance_buffer_m"],
        spec["max_braking"],
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
