"""Analyze the AEBS recoverable domain before training another barrier."""

import argparse
import json
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

from Aebs.mvp.robust_sbc import discrete_stopping_distance, verifier_grid_states
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller


def worst_endpoint_actions(controller, distance_m, speed, radii, distance_scale):
    low_distance = np.clip(distance_m / distance_scale - radii, 5.0 / distance_scale, 16.0 / distance_scale)
    high_distance = np.clip(distance_m / distance_scale + radii, 5.0 / distance_scale, 16.0 / distance_scale)
    low = np.stack((low_distance, speed), axis=1).astype(np.float32)
    high = np.stack((high_distance, speed), axis=1).astype(np.float32)
    low_action, _ = controller.predict(low, deterministic=True)
    high_action, _ = controller.predict(high, deterministic=True)
    low_action = np.asarray(low_action, dtype=np.float32).reshape(-1)
    high_action = np.asarray(high_action, dtype=np.float32).reshape(-1)
    return np.minimum(low_action, high_action)


def analyze(
    controller_path: str,
    semantic_checkpoint: str,
    data_path: str,
    output_dir: Path,
    grid_size: int,
    max_steps: int,
    safety_distance_m: float,
    unsafe_speed: float,
    max_braking: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        distance_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, distance_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")
    controller = load_controller(controller_path)
    states = verifier_grid_states(distance_scale, grid_size)
    initial_distance = states[:, 0] * distance_scale
    initial_speed = states[:, 1]

    stopping_distance = np.maximum(initial_speed**2 - unsafe_speed**2, 0.0) / (2.0 * max_braking)
    recoverability_margin = initial_distance - safety_distance_m - stopping_distance
    discrete_distance = discrete_stopping_distance(initial_speed, unsafe_speed, max_braking, 0.05)
    discrete_margin = initial_distance - safety_distance_m - discrete_distance
    initially_goal = (initial_distance <= safety_distance_m) & (initial_speed <= unsafe_speed)
    initially_unsafe = (initial_distance <= safety_distance_m) & (initial_speed > unsafe_speed)
    stopped_initially = initial_speed <= 0.0
    operational = ~(initially_goal | initially_unsafe | stopped_initially)
    physically_unrecoverable = (recoverability_margin < 0.0) & operational
    discrete_unrecoverable = (discrete_margin < 0.0) & operational

    distance = initial_distance.copy()
    speed = initial_speed.copy()
    active = ~(initially_goal | initially_unsafe | (speed <= 0.0))
    reached_goal = initially_goal.copy()
    reached_unsafe = initially_unsafe.copy()
    stopped_safe = (speed <= 0.0) & ~initially_unsafe
    steps = np.zeros(len(states), dtype=np.int64)
    for _ in range(max_steps):
        indices = np.flatnonzero(active)
        if len(indices) == 0:
            break
        d = distance[indices]
        v = speed[indices]
        radii = contract.radius(d).astype(np.float32)
        action = worst_endpoint_actions(controller, d, v, radii, distance_scale)
        next_distance = d - v * 0.05
        next_speed = np.clip(v - action * 0.05, 0.0, 3.0)
        distance[indices] = next_distance
        speed[indices] = next_speed
        steps[indices] += 1

        unsafe_now = (next_distance <= safety_distance_m) & (next_speed > unsafe_speed)
        goal_now = (next_distance <= safety_distance_m) & (next_speed <= unsafe_speed)
        stopped_now = next_speed <= 0.0
        outside_now = (next_distance <= 5.0) | (next_distance >= 16.0)
        reached_unsafe[indices[unsafe_now]] = True
        reached_goal[indices[goal_now & ~unsafe_now]] = True
        stopped_safe[indices[stopped_now & ~unsafe_now]] = True
        done = unsafe_now | goal_now | stopped_now | outside_now
        active[indices[done]] = False

    unresolved = active.copy()
    rollout_unsafe_from_operational = reached_unsafe & operational
    continuous_false_safe = (~physically_unrecoverable) & rollout_unsafe_from_operational
    discrete_false_safe = (~discrete_unrecoverable) & rollout_unsafe_from_operational
    continuous_conservative = physically_unrecoverable & ~rollout_unsafe_from_operational
    discrete_conservative = discrete_unrecoverable & ~rollout_unsafe_from_operational
    metrics = {
        "experiment": "aebs_certificate_domain_analysis",
        "controller": controller_path,
        "grid_size": int(grid_size),
        "grid_states": int(len(states)),
        "max_steps": int(max_steps),
        "physical_recoverability": {
            "operational_count": int(np.sum(operational)),
            "recoverable_count": int(np.sum(operational & ~physically_unrecoverable)),
            "unrecoverable_count": int(np.sum(physically_unrecoverable)),
            "minimum_margin_m": float(np.min(recoverability_margin[operational])),
            "initial_region_minimum_margin_m": float(
                15.0 - safety_distance_m - (3.0**2 - unsafe_speed**2) / (2.0 * max_braking)
            ),
        },
        "discrete_recoverability": {
            "operational_count": int(np.sum(operational)),
            "recoverable_count": int(np.sum(operational & ~discrete_unrecoverable)),
            "unrecoverable_count": int(np.sum(discrete_unrecoverable)),
            "minimum_margin_m": float(np.min(discrete_margin[operational])),
            "initial_region_minimum_margin_m": float(
                15.0
                - safety_distance_m
                - discrete_stopping_distance(np.array([3.0]), unsafe_speed, max_braking, 0.05)[0]
            ),
        },
        "worst_endpoint_rollout": {
            "goal_count": int(np.sum(reached_goal)),
            "stopped_safe_count": int(np.sum(stopped_safe & ~reached_goal)),
            "unsafe_count": int(np.sum(reached_unsafe)),
            "unresolved_count": int(np.sum(unresolved)),
            "maximum_steps": int(np.max(steps)),
        },
        "comparison": {
            "rollout_unsafe_from_operational": int(np.sum(rollout_unsafe_from_operational)),
            "continuous_recoverable_but_rollout_unsafe": int(np.sum(continuous_false_safe)),
            "continuous_unrecoverable_but_rollout_not_unsafe": int(np.sum(continuous_conservative)),
            "discrete_recoverable_but_rollout_unsafe": int(np.sum(discrete_false_safe)),
            "discrete_unrecoverable_but_rollout_not_unsafe": int(np.sum(discrete_conservative)),
            "max_discrete_margin_among_rollout_unsafe_m": (
                float(np.max(discrete_margin[rollout_unsafe_from_operational]))
                if np.any(rollout_unsafe_from_operational)
                else None
            ),
        },
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", default="results/mvp/03_domain_analysis")
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--safety-distance-m", type=float, default=6.0)
    parser.add_argument("--unsafe-speed", type=float, default=0.5)
    parser.add_argument("--max-braking", type=float, default=3.0)
    args = parser.parse_args()
    metrics = analyze(
        args.controller,
        args.semantic_checkpoint,
        args.data,
        Path(args.output_dir),
        args.grid_size,
        args.max_steps,
        args.safety_distance_m,
        args.unsafe_speed,
        args.max_braking,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
