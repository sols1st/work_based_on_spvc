"""Dense local refinement of the teacher-to-student safety diagnostic.

The first 80x80 check found a thin sampled margin near d=6--8 m and v=2--3
m/s.  This script refines only that region, streams controller evaluations in
small chunks, and reports both distance-to-viability and action-threshold
margins.  It remains a sampling diagnostic rather than a formal NN bound.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from Aebs.mvp.check_certificate_transfer import (
    certificate_masks_for_check,
    minimum_safe_action,
    next_recoverability_margin,
    predict_actions,
    summarize,
)
from Aebs.semantic.robust_controller import load_contract
from Aebs.semantic.safety_filter import load_controller


def chunks(length: int, size: int) -> Iterable[slice]:
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def dense_region_states(
    distance_scale: float,
    distance_low_m: float,
    distance_high_m: float,
    speed_low: float,
    speed_high: float,
    distance_points: int,
    speed_points: int,
) -> np.ndarray:
    distance = np.linspace(distance_low_m, distance_high_m, distance_points)
    speed = np.linspace(speed_low, speed_high, speed_points)
    mesh_d, mesh_v = np.meshgrid(distance, speed)
    return np.stack(
        (mesh_d.reshape(-1) / distance_scale, mesh_v.reshape(-1)), axis=1
    ).astype(np.float32)


def reserve_counts(values: np.ndarray) -> Dict[str, Dict[str, float]]:
    result = {}
    for reserve_m in (0.0, 0.001, 0.005, 0.01, 0.05, 0.1):
        passed = values >= reserve_m
        result[f"{reserve_m:.3f}"] = {
            "count": int(np.sum(passed)),
            "rate": float(np.mean(passed)),
        }
    return result


def run_refinement(
    teacher_path: str,
    student_path: str,
    semantic_checkpoint: str,
    output_dir: Path,
    distance_low_m: float,
    distance_high_m: float,
    speed_low: float,
    speed_high: float,
    distance_points: int,
    speed_points: int,
    semantic_samples: int,
    state_chunk_size: int,
    controller_batch_size: int,
    tolerance: float,
) -> Dict:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    teacher = load_controller(teacher_path)
    student = load_controller(student_path)

    all_states = dense_region_states(
        distance_scale,
        distance_low_m,
        distance_high_m,
        speed_low,
        speed_high,
        distance_points,
        speed_points,
    )
    masks = certificate_masks_for_check(all_states, distance_scale)
    state_indices = np.flatnonzero(masks["decrease"])
    states = all_states[state_indices]
    distance_m = states[:, 0].astype(np.float64) * distance_scale
    speed = states[:, 1].astype(np.float64)
    radius_norm = contract.radius(distance_m).astype(np.float64)
    multipliers = np.linspace(-1.0, 1.0, semantic_samples, dtype=np.float64)
    required_action = minimum_safe_action(distance_m, speed, 6.0, 0.5, 3.0)

    count = len(states)
    teacher_next_margin = np.full(count, np.inf, dtype=np.float64)
    student_next_margin = np.full(count, np.inf, dtype=np.float64)
    teacher_action_margin = np.full(count, np.inf, dtype=np.float64)
    student_action_margin = np.full(count, np.inf, dtype=np.float64)
    worst_underbraking = np.full(count, -np.inf, dtype=np.float64)
    worst_absolute_action_error = np.zeros(count, dtype=np.float64)
    worst_student_multiplier = np.zeros(count, dtype=np.float64)
    worst_student_action = np.zeros(count, dtype=np.float64)
    worst_teacher_action_at_student_case = np.zeros(count, dtype=np.float64)

    for selection in chunks(count, state_chunk_size):
        local_states = states[selection]
        local_distance = distance_m[selection]
        local_speed = speed[selection]
        observed_distance = (
            local_states[:, 0, None].astype(np.float64)
            + radius_norm[selection, None] * multipliers[None, :]
        )
        observed_distance = np.clip(
            observed_distance, 5.0 / distance_scale, 16.0 / distance_scale
        )
        observations = np.stack(
            (
                observed_distance,
                np.broadcast_to(local_speed[:, None], observed_distance.shape),
            ),
            axis=2,
        ).reshape(-1, 2)
        local_count = len(local_states)
        teacher_action = predict_actions(
            teacher, observations, controller_batch_size
        ).reshape(local_count, semantic_samples)
        student_action = predict_actions(
            student, observations, controller_batch_size
        ).reshape(local_count, semantic_samples)
        teacher_margin = next_recoverability_margin(
            local_distance[:, None],
            local_speed[:, None],
            teacher_action,
            6.0,
            0.5,
            3.0,
        )
        student_margin = next_recoverability_margin(
            local_distance[:, None],
            local_speed[:, None],
            student_action,
            6.0,
            0.5,
            3.0,
        )
        local_worst_index = np.argmin(student_margin, axis=1)
        rows = np.arange(local_count)

        teacher_next_margin[selection] = np.min(teacher_margin, axis=1)
        student_next_margin[selection] = student_margin[rows, local_worst_index]
        teacher_action_margin[selection] = np.min(
            teacher_action - required_action[selection, None], axis=1
        )
        student_action_margin[selection] = np.min(
            student_action - required_action[selection, None], axis=1
        )
        difference = teacher_action - student_action
        worst_underbraking[selection] = np.max(difference, axis=1)
        worst_absolute_action_error[selection] = np.max(
            np.abs(difference), axis=1
        )
        worst_student_multiplier[selection] = multipliers[local_worst_index]
        worst_student_action[selection] = student_action[rows, local_worst_index]
        worst_teacher_action_at_student_case[selection] = teacher_action[
            rows, local_worst_index
        ]

    teacher_pass = teacher_next_margin >= -tolerance
    student_pass = student_next_margin >= -tolerance
    if not np.all(teacher_pass):
        decision = "teacher_counterexample_found"
    elif not np.all(student_pass):
        decision = "student_counterexample_found"
    elif float(np.min(student_action_margin)) <= 0.0:
        decision = "sampled_pass_but_action_threshold_not_strict"
    else:
        decision = "dense_sampled_pass_ready_for_nn_bounds"

    order = np.argsort(student_next_margin)
    worst_states = []
    for index in order[: min(20, count)]:
        worst_states.append(
            {
                "distance_m": float(distance_m[index]),
                "speed_m_per_s": float(speed[index]),
                "semantic_radius_m": float(radius_norm[index] * distance_scale),
                "worst_semantic_multiplier": float(worst_student_multiplier[index]),
                "required_action": float(required_action[index]),
                "student_action": float(worst_student_action[index]),
                "teacher_action": float(worst_teacher_action_at_student_case[index]),
                "student_action_margin": float(student_action_margin[index]),
                "student_next_margin_m": float(student_next_margin[index]),
                "teacher_next_margin_m": float(teacher_next_margin[index]),
            }
        )

    metrics = {
        "experiment": "dense_local_teacher_student_transfer_refinement",
        "status": decision,
        "formal_guarantee": False,
        "validity_note": (
            "This is a dense local sampling refinement. It does not bound the "
            "student network between semantic samples or states between grid points."
        ),
        "semantic_checkpoint": semantic_checkpoint,
        "region": {
            "distance_low_m": float(distance_low_m),
            "distance_high_m": float(distance_high_m),
            "speed_low_m_per_s": float(speed_low),
            "speed_high_m_per_s": float(speed_high),
        },
        "resolution": {
            "distance_points": int(distance_points),
            "speed_points": int(speed_points),
            "distance_spacing_m": float(
                (distance_high_m - distance_low_m) / (distance_points - 1)
            ),
            "speed_spacing_m_per_s": float(
                (speed_high - speed_low) / (speed_points - 1)
            ),
            "semantic_samples_per_state": int(semantic_samples),
        },
        "region_grid_states": int(len(all_states)),
        "checked_recoverable_operational_states": int(count),
        "checked_state_observation_pairs": int(count * semantic_samples),
        "teacher": {
            "path": teacher_path,
            "pass_count": int(np.sum(teacher_pass)),
            "violation_count": int(np.sum(~teacher_pass)),
            "pass_rate": float(np.mean(teacher_pass)),
            "next_recoverability_margin_m": summarize(teacher_next_margin),
            "action_surplus_over_required": summarize(teacher_action_margin),
        },
        "student": {
            "path": student_path,
            "pass_count": int(np.sum(student_pass)),
            "violation_count": int(np.sum(~student_pass)),
            "pass_rate": float(np.mean(student_pass)),
            "next_recoverability_margin_m": summarize(student_next_margin),
            "next_margin_reserve_rates": reserve_counts(student_next_margin),
            "action_surplus_over_required": summarize(student_action_margin),
        },
        "transfer": {
            "worst_underbraking": summarize(worst_underbraking),
            "worst_absolute_action_error": summarize(worst_absolute_action_error),
        },
        "worst_states": worst_states,
        "runtime_seconds": float(time.time() - started),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output_dir / "state_diagnostics.npz",
        state_indices=state_indices,
        states=states,
        distance_m=distance_m,
        speed=speed,
        required_action=required_action,
        teacher_next_margin_m=teacher_next_margin,
        student_next_margin_m=student_next_margin,
        teacher_action_margin=teacher_action_margin,
        student_action_margin=student_action_margin,
        worst_underbraking=worst_underbraking,
        worst_absolute_action_error=worst_absolute_action_error,
        worst_student_multiplier=worst_student_multiplier,
    )

    print("[Dense local certificate-transfer refinement]")
    print("level: dense sampled diagnostic (NOT a formal guarantee)")
    print(f"decision: {decision}")
    print(
        f"region: d=[{distance_low_m:.3f}, {distance_high_m:.3f}] m, "
        f"v=[{speed_low:.3f}, {speed_high:.3f}] m/s"
    )
    print(
        f"resolution: {distance_points}x{speed_points} states, "
        f"{semantic_samples} semantic samples/state"
    )
    print(
        f"checked: states={count}, pairs={count * semantic_samples}"
    )
    print(
        f"teacher: pass={100.0 * np.mean(teacher_pass):.2f}%, "
        f"violations={np.sum(~teacher_pass)}, "
        f"min next margin={np.min(teacher_next_margin):.9f} m"
    )
    print(
        f"student: pass={100.0 * np.mean(student_pass):.2f}%, "
        f"violations={np.sum(~student_pass)}, "
        f"min next margin={np.min(student_next_margin):.9f} m, "
        f"min action surplus={np.min(student_action_margin):.9f}"
    )
    worst = worst_states[0]
    print(
        "worst student state: "
        f"d={worst['distance_m']:.6f} m, "
        f"v={worst['speed_m_per_s']:.6f} m/s, "
        f"semantic_multiplier={worst['worst_semantic_multiplier']:.3f}, "
        f"margin={worst['student_next_margin_m']:.9f} m"
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
        default="results/mvp/06_certificate_transfer_local_refinement",
    )
    parser.add_argument("--distance-low-m", type=float, default=6.0)
    parser.add_argument("--distance-high-m", type=float, default=8.0)
    parser.add_argument("--speed-low", type=float, default=2.0)
    parser.add_argument("--speed-high", type=float, default=3.0)
    parser.add_argument("--distance-points", type=int, default=201)
    parser.add_argument("--speed-points", type=int, default=201)
    parser.add_argument("--semantic-samples", type=int, default=129)
    parser.add_argument("--state-chunk-size", type=int, default=512)
    parser.add_argument("--controller-batch-size", type=int, default=65536)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()
    if args.distance_points < 2 or args.speed_points < 2:
        raise ValueError("distance-points and speed-points must be at least 2")
    if args.semantic_samples < 2:
        raise ValueError("semantic-samples must be at least 2")
    run_refinement(
        args.teacher,
        args.student,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.distance_low_m,
        args.distance_high_m,
        args.speed_low,
        args.speed_high,
        args.distance_points,
        args.speed_points,
        args.semantic_samples,
        args.state_chunk_size,
        args.controller_batch_size,
        args.tolerance,
    )


if __name__ == "__main__":
    main()
