"""Collect the minimal state-dependent disturbance boxes used by the MVP."""

import argparse
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.semantic.robust_controller import action, load_contract
from Aebs.semantic.safety_filter import load_controller
from Aebs.uncertainty.model import StateDependentUncertainty


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def next_state(distance_m: float, speed: float, acceleration: float, distance_scale: float) -> np.ndarray:
    next_distance_m = distance_m - speed * 0.05
    next_speed = float(np.clip(speed - acceleration * 0.05, 0.0, 3.0))
    return np.array([next_distance_m / distance_scale, next_speed], dtype=np.float64)


def collect_deterministic_process_uncertainty(
    data_path: str,
    output_dir: Path,
    seed: int,
    bins: int,
    states_per_cell: int,
) -> dict:
    """Record zero process residual for the deterministic MVP simulator.

    Semantic action error is handled separately by the endpoint enumeration in the
    robust SBC and must not be inserted into this process-disturbance model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "r") as stream:
        float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    distance_edges = np.linspace(5.0, 16.0, bins + 1)
    speed_edges = np.linspace(0.0, 3.0, bins + 1)
    zeros = np.zeros((bins, bins, 2), dtype=np.float64)
    counts = np.full((bins, bins), states_per_cell, dtype=np.int64)
    uncertainty = StateDependentUncertainty(
        distance_edges_m=distance_edges,
        speed_edges=speed_edges,
        mean=zeros.copy(),
        support_low=zeros.copy(),
        support_high=zeros.copy(),
        mean_radius=zeros.copy(),
        counts=counts,
    )
    uncertainty.to_npz(str(output_dir / "state_dependent_uncertainty.npz"))
    metrics = {
        "experiment": "state_dependent_process_uncertainty_mvp",
        "source": "deterministic_simulator_residual",
        "seed": int(seed),
        "bins": int(bins),
        "states_per_cell": int(states_per_cell),
        "cells": int(bins * bins),
        "min_count_per_cell": int(counts.min()),
        "max_count_per_cell": int(counts.max()),
        "mean_radius_max": [0.0, 0.0],
        "support_low_min": [0.0, 0.0],
        "support_high_max": [0.0, 0.0],
        "note": "The current AEBS simulator is deterministic; real process residual data is not present in Downsampled.h5.",
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def collect_uncertainty(
    controller: PPO,
    semantic_checkpoint: str,
    data_path: str,
    output_dir: Path,
    seed: int,
    bins: int,
    states_per_cell: int,
    errors_per_state: int,
    alpha: float,
) -> dict:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        dataset_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, dataset_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")

    rng = np.random.default_rng(seed)
    distance_edges = np.linspace(5.0, 16.0, bins + 1)
    speed_edges = np.linspace(0.0, 3.0, bins + 1)
    samples = [[[] for _ in range(bins)] for _ in range(bins)]
    started = time.time()

    for d_cell in range(bins):
        for v_cell in range(bins):
            d_low, d_high = distance_edges[d_cell], distance_edges[d_cell + 1]
            v_low, v_high = speed_edges[v_cell], speed_edges[v_cell + 1]
            for _ in range(states_per_cell):
                distance_m = float(rng.uniform(d_low, d_high))
                speed = float(rng.uniform(v_low, v_high))
                distance_norm = distance_m / dataset_scale
                nominal_action = action(controller, distance_norm, speed)
                nominal_next = next_state(distance_m, speed, nominal_action, dataset_scale)
                radius = float(contract.radius(np.array([distance_m]))[0])
                semantic_errors = rng.uniform(-radius, radius, size=errors_per_state)
                semantic_errors[0] = -radius
                if errors_per_state > 1:
                    semantic_errors[1] = radius
                for error in semantic_errors:
                    perturbed_action = action(controller, distance_norm + float(error), speed)
                    perturbed_next = next_state(distance_m, speed, perturbed_action, dataset_scale)
                    samples[d_cell][v_cell].append(perturbed_next - nominal_next)

    counts = np.zeros((bins, bins), dtype=np.int64)
    mean = np.zeros((bins, bins, 2), dtype=np.float64)
    support_low = np.zeros((bins, bins, 2), dtype=np.float64)
    support_high = np.zeros((bins, bins, 2), dtype=np.float64)
    mean_radius = np.zeros((bins, bins, 2), dtype=np.float64)
    for d_cell in range(bins):
        for v_cell in range(bins):
            values = np.asarray(samples[d_cell][v_cell], dtype=np.float64)
            counts[d_cell, v_cell] = len(values)
            mean[d_cell, v_cell] = values.mean(axis=0)
            empirical_low = values.min(axis=0)
            empirical_high = values.max(axis=0)
            ranges = np.maximum(empirical_high - empirical_low, 1e-12)
            radius = ranges * np.sqrt(np.log(2.0 / alpha) / (2.0 * len(values)))
            mean_radius[d_cell, v_cell] = radius
            support_low[d_cell, v_cell] = empirical_low - radius
            support_high[d_cell, v_cell] = empirical_high + radius

    uncertainty = StateDependentUncertainty(
        distance_edges_m=distance_edges,
        speed_edges=speed_edges,
        mean=mean,
        support_low=support_low,
        support_high=support_high,
        mean_radius=mean_radius,
        counts=counts,
    )
    uncertainty.to_npz(str(output_dir / "state_dependent_uncertainty.npz"))
    metrics = {
        "experiment": "state_dependent_uncertainty_mvp",
        "seed": seed,
        "bins": bins,
        "states_per_cell": states_per_cell,
        "errors_per_state": errors_per_state,
        "cells": int(bins * bins),
        "min_count_per_cell": int(counts.min()),
        "max_count_per_cell": int(counts.max()),
        "mean_radius_max": mean_radius.max(axis=(0, 1)).tolist(),
        "support_low_min": support_low.min(axis=(0, 1)).tolist(),
        "support_high_max": support_high.max(axis=(0, 1)).tolist(),
        "runtime_seconds": float(time.time() - started),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("deterministic_process_residual", "legacy_semantic_action_delta"),
        default="deterministic_process_residual",
    )
    parser.add_argument("--semantic-checkpoint")
    parser.add_argument("--controller")
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", default="results/mvp/03_uncertainty")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--states-per-cell", type=int, default=256)
    parser.add_argument("--errors-per-state", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if args.source == "deterministic_process_residual":
        metrics = collect_deterministic_process_uncertainty(
            args.data,
            Path(args.output_dir),
            args.seed,
            args.bins,
            args.states_per_cell,
        )
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return
    if not args.semantic_checkpoint or not args.controller:
        parser.error("legacy_semantic_action_delta requires --semantic-checkpoint and --controller")
    controller = load_controller(args.controller)
    metrics = collect_uncertainty(
        controller,
        args.semantic_checkpoint,
        args.data,
        Path(args.output_dir),
        args.seed,
        args.bins,
        args.states_per_cell,
        args.errors_per_state,
        args.alpha,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
