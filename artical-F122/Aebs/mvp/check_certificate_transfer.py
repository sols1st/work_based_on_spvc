"""Fast go/no-go diagnostic for safety transfer from a filter to a standalone PPO.

This is a dense fixed-grid and semantic-sampling diagnostic.  It deliberately
does not claim a continuous-state formal certificate: controller extrema
between sampled semantic inputs and state-grid points are not bounded here.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from Aebs.mvp.robust_sbc import (
    certificate_masks,
    discrete_stopping_distance,
    verifier_grid_states,
)
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller


DT = 0.05


def predict_actions(controller, observations: np.ndarray, batch_size: int) -> np.ndarray:
    """Run a PPO or filter controller in bounded-memory batches."""

    outputs = []
    for start in range(0, len(observations), batch_size):
        actions, _ = controller.predict(
            observations[start : start + batch_size].astype(np.float32),
            deterministic=True,
        )
        outputs.append(np.asarray(actions, dtype=np.float64).reshape(-1))
    return np.clip(np.concatenate(outputs), -3.0, 3.0)


def next_recoverability_margin(
    distance_m: np.ndarray,
    speed: np.ndarray,
    action: np.ndarray,
    safety_distance_m: float,
    unsafe_speed: float,
    max_braking: float,
) -> np.ndarray:
    """Discrete stopping-distance margin after applying one action."""

    next_distance = distance_m - speed * DT
    next_speed = np.clip(speed - action * DT, 0.0, 3.0)
    remaining = discrete_stopping_distance(
        next_speed, unsafe_speed, max_braking, DT
    )
    return next_distance - safety_distance_m - remaining


def minimum_safe_action(
    distance_m: np.ndarray,
    speed: np.ndarray,
    safety_distance_m: float,
    unsafe_speed: float,
    max_braking: float,
) -> np.ndarray:
    """Smallest current action whose successor remains discretely recoverable.

    The AEBS dynamics are monotone in the braking action.  Bisection therefore
    finds the threshold action on each state without fitting another network.
    """

    low = np.full_like(distance_m, -3.0, dtype=np.float64)
    high = np.full_like(distance_m, max_braking, dtype=np.float64)
    low_is_safe = next_recoverability_margin(
        distance_m,
        speed,
        low,
        safety_distance_m,
        unsafe_speed,
        max_braking,
    ) >= 0.0
    for _ in range(45):
        middle = 0.5 * (low + high)
        middle_is_safe = next_recoverability_margin(
            distance_m,
            speed,
            middle,
            safety_distance_m,
            unsafe_speed,
            max_braking,
        ) >= 0.0
        high = np.where(middle_is_safe, middle, high)
        low = np.where(middle_is_safe, low, middle)
    return np.where(low_is_safe, -3.0, high)


def summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
    }


def certificate_masks_for_check(states: np.ndarray, distance_scale: float) -> Dict[str, np.ndarray]:
    return certificate_masks(
        states,
        distance_scale,
        terminal_speed_threshold=0.5,
        goal_distance_m=6.0,
        goal_speed=0.5,
        unsafe_distance_low_m=5.0,
        unsafe_distance_high_m=6.0,
        unsafe_speed=0.5,
        use_recoverable_domain=True,
        max_braking=3.0,
    )


def run_check(
    teacher_path: str,
    student_path: str,
    semantic_checkpoint: str,
    output_dir: Path,
    grid_size: int,
    semantic_samples: int,
    batch_size: int,
    tolerance: float,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    teacher = load_controller(teacher_path)
    student = load_controller(student_path)

    all_states = verifier_grid_states(distance_scale, grid_size)
    masks = certificate_masks_for_check(all_states, distance_scale)
    state_indices = np.flatnonzero(masks["decrease"])
    states = all_states[state_indices]
    true_distance_m = states[:, 0].astype(np.float64) * distance_scale
    speed = states[:, 1].astype(np.float64)

    radii_norm = contract.radius(true_distance_m).astype(np.float64)
    multipliers = np.linspace(-1.0, 1.0, semantic_samples, dtype=np.float64)
    observed_distance_norm = (
        states[:, 0, None].astype(np.float64)
        + radii_norm[:, None] * multipliers[None, :]
    )
    observed_distance_norm = np.clip(
        observed_distance_norm, 5.0 / distance_scale, 16.0 / distance_scale
    )
    observations = np.stack(
        (
            observed_distance_norm,
            np.broadcast_to(speed[:, None], observed_distance_norm.shape),
        ),
        axis=2,
    ).reshape(-1, 2)

    teacher_action = predict_actions(teacher, observations, batch_size).reshape(
        len(states), semantic_samples
    )
    student_action = predict_actions(student, observations, batch_size).reshape(
        len(states), semantic_samples
    )
    required_action = minimum_safe_action(
        true_distance_m,
        speed,
        safety_distance_m=6.0,
        unsafe_speed=0.5,
        max_braking=3.0,
    )

    teacher_action_margin = np.min(
        teacher_action - required_action[:, None], axis=1
    )
    student_action_margin = np.min(
        student_action - required_action[:, None], axis=1
    )
    teacher_next_margin = np.min(
        next_recoverability_margin(
            true_distance_m[:, None],
            speed[:, None],
            teacher_action,
            6.0,
            0.5,
            3.0,
        ),
        axis=1,
    )
    student_next_margin = np.min(
        next_recoverability_margin(
            true_distance_m[:, None],
            speed[:, None],
            student_action,
            6.0,
            0.5,
            3.0,
        ),
        axis=1,
    )
    worst_underbraking = np.max(teacher_action - student_action, axis=1)
    worst_absolute_action_error = np.max(
        np.abs(teacher_action - student_action), axis=1
    )

    teacher_pass = teacher_next_margin >= -tolerance
    student_pass = student_next_margin >= -tolerance
    transfer_pass = teacher_pass & student_pass
    teacher_pass_rate = float(np.mean(teacher_pass))
    student_pass_rate = float(np.mean(student_pass))

    if teacher_pass_rate < 1.0:
        decision = "teacher_not_certifiable_on_sampled_domain"
    elif student_pass_rate >= 0.90:
        decision = "promising_for_local_formalization"
    elif student_pass_rate >= 0.50:
        decision = "borderline_local_student_repair_needed"
    else:
        decision = "not_ready_student_or_domain_must_change"

    distance_edges = np.asarray(contract.bin_edges, dtype=np.float64)
    speed_edges = np.linspace(0.0, 3.0, 5)
    distance_bin = np.clip(
        np.digitize(true_distance_m, distance_edges[1:-1]), 0, 3
    )
    speed_bin = np.clip(np.digitize(speed, speed_edges[1:-1]), 0, 3)
    cells = []
    for d_bin in range(4):
        for v_bin in range(4):
            selected = (distance_bin == d_bin) & (speed_bin == v_bin)
            if not np.any(selected):
                continue
            cells.append(
                {
                    "distance_bin": d_bin,
                    "speed_bin": v_bin,
                    "count": int(np.sum(selected)),
                    "teacher_pass_rate": float(np.mean(teacher_pass[selected])),
                    "student_pass_rate": float(np.mean(student_pass[selected])),
                    "minimum_student_next_margin_m": float(
                        np.min(student_next_margin[selected])
                    ),
                    "maximum_underbraking": float(
                        np.max(worst_underbraking[selected])
                    ),
                }
            )

    worst_index = int(np.argmin(student_next_margin))
    metrics = {
        "experiment": "sampled_teacher_student_certificate_transfer_check",
        "status": decision,
        "formal_guarantee": False,
        "validity_note": (
            "Dense fixed-grid and semantic-input sampling is a go/no-go diagnostic, "
            "not a continuous-state proof. A formal result still needs sound neural "
            "output bounds between samples and a cell-discretization remainder. "
            "The checked property is one-step invariance of the discrete recoverable "
            "set; task completion/liveness remains a separate rollout property."
        ),
        "semantic_checkpoint": semantic_checkpoint,
        "grid_size": int(grid_size),
        "all_grid_states": int(len(all_states)),
        "checked_recoverable_operational_states": int(len(states)),
        "semantic_samples_per_state": int(semantic_samples),
        "checked_state_observation_pairs": int(len(observations)),
        "tolerance_m": float(tolerance),
        "teacher": {
            "path": teacher_path,
            "pass_count": int(np.sum(teacher_pass)),
            "violation_count": int(np.sum(~teacher_pass)),
            "pass_rate": teacher_pass_rate,
            "next_recoverability_margin_m": summarize(teacher_next_margin),
            "action_surplus_over_required": summarize(teacher_action_margin),
        },
        "student": {
            "path": student_path,
            "pass_count": int(np.sum(student_pass)),
            "violation_count": int(np.sum(~student_pass)),
            "pass_rate": student_pass_rate,
            "next_recoverability_margin_m": summarize(student_next_margin),
            "action_surplus_over_required": summarize(student_action_margin),
        },
        "transfer": {
            "pass_count": int(np.sum(transfer_pass)),
            "violation_count": int(np.sum(~transfer_pass)),
            "pass_rate": float(np.mean(transfer_pass)),
            "worst_underbraking": summarize(worst_underbraking),
            "worst_absolute_action_error": summarize(worst_absolute_action_error),
        },
        "worst_student_state": {
            "distance_m": float(true_distance_m[worst_index]),
            "speed_m_per_s": float(speed[worst_index]),
            "semantic_radius_m": float(radii_norm[worst_index] * distance_scale),
            "minimum_next_recoverability_margin_m": float(
                student_next_margin[worst_index]
            ),
            "minimum_action_surplus": float(student_action_margin[worst_index]),
            "maximum_underbraking": float(worst_underbraking[worst_index]),
        },
        "cells": cells,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "state_diagnostics.npz",
        state_indices=state_indices,
        states=states,
        true_distance_m=true_distance_m,
        speed=speed,
        required_action=required_action,
        teacher_next_margin_m=teacher_next_margin,
        student_next_margin_m=student_next_margin,
        teacher_action_margin=teacher_action_margin,
        student_action_margin=student_action_margin,
        worst_underbraking=worst_underbraking,
        worst_absolute_action_error=worst_absolute_action_error,
    )

    print("[Teacher -> standalone PPO safety transfer check]")
    print("level: sampled grid diagnostic (NOT a formal guarantee)")
    print(f"decision: {decision}")
    print(
        f"states: {len(states)}, semantic samples/state: {semantic_samples}, "
        f"pairs: {len(observations)}"
    )
    print(
        f"teacher: pass={100.0 * teacher_pass_rate:.2f}%, "
        f"violations={np.sum(~teacher_pass)}, "
        f"min next margin={np.min(teacher_next_margin):.6f} m"
    )
    print(
        f"student: pass={100.0 * student_pass_rate:.2f}%, "
        f"violations={np.sum(~student_pass)}, "
        f"min next margin={np.min(student_next_margin):.6f} m"
    )
    print(
        f"action: max underbraking={np.max(worst_underbraking):.6f}, "
        f"max abs error={np.max(worst_absolute_action_error):.6f}"
    )
    print(
        "worst student state: "
        f"d={true_distance_m[worst_index]:.3f} m, "
        f"v={speed[worst_index]:.3f} m/s, "
        f"margin={student_next_margin[worst_index]:.6f} m"
    )
    print(f"metrics: {output_dir / 'metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        default=(
            "results/mvp/02_safety_filter_anti_stall_compare/"
            "adaptive_anti_stall_filter.json"
        ),
    )
    parser.add_argument(
        "--student",
        default="results/mvp/02_standalone_ppo_distilled_v2/standalone_ppo.zip",
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/05_certificate_transfer_check",
    )
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--semantic-samples", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.grid_size < 2:
        raise ValueError("grid-size must be at least 2")
    if args.semantic_samples < 2:
        raise ValueError("semantic-samples must be at least 2")
    run_check(
        args.teacher,
        args.student,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.grid_size,
        args.semantic_samples,
        args.batch_size,
        args.tolerance,
    )


if __name__ == "__main__":
    main()
