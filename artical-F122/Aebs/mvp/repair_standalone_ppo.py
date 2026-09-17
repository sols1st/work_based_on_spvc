"""Minimal local repair of the filter-free PPO after dense counterexamples.

The successful v2 PPO is kept as an anchor on the original full-domain grid.
Only states whose sampled action surplus is small are relabelled with the safe
teacher/required-braking target.  Deployment and all evaluations use the
repaired PPO alone; the filter is never called at runtime.
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from Aebs.mvp.refine_certificate_transfer import run_refinement
from Aebs.semantic.distill_filter_to_ppo import (
    MODES,
    actor_output,
    observation_grid,
    policy_action_metrics,
)
from Aebs.semantic.robust_controller import evaluate, load_contract
from Aebs.semantic.safety_filter import load_controller


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def local_repair_dataset(
    diagnostics_path: str,
    contract,
    distance_scale: float,
    teacher,
    selection_margin: float,
    semantic_samples: int,
    target_margin: float,
    raw_target_headroom: float,
) -> Dict[str, np.ndarray]:
    diagnostics = np.load(diagnostics_path)
    states = np.asarray(diagnostics["states"], dtype=np.float32)
    required_action = np.asarray(
        diagnostics["required_action"], dtype=np.float32
    )
    old_margin = np.asarray(
        diagnostics["student_action_margin"], dtype=np.float32
    )
    selected = old_margin < selection_margin
    if not np.any(selected):
        raise ValueError(
            f"no states have student_action_margin < {selection_margin}"
        )

    states = states[selected]
    required_action = required_action[selected]
    old_margin = old_margin[selected]
    distance_m = states[:, 0] * distance_scale
    radius_norm = contract.radius(distance_m).astype(np.float32)
    multipliers = np.linspace(-1.0, 1.0, semantic_samples, dtype=np.float32)
    observed_distance = states[:, 0, None] + radius_norm[:, None] * multipliers
    observed_distance = np.clip(
        observed_distance, 5.0 / distance_scale, 16.0 / distance_scale
    )
    observations = np.stack(
        (
            observed_distance,
            np.broadcast_to(states[:, 1, None], observed_distance.shape),
        ),
        axis=2,
    ).reshape(-1, 2).astype(np.float32)
    teacher_action, _ = teacher.predict(observations, deterministic=True)
    teacher_action = np.asarray(teacher_action, dtype=np.float32).reshape(-1, 1)
    repeated_required = np.repeat(required_action, semantic_samples).reshape(-1, 1)
    safety_target = np.maximum(
        teacher_action, repeated_required + float(target_margin)
    )
    # A smooth actor trained to a target of exactly 3.0 may approach the action
    # limit from below forever.  For critical maximum-braking samples, a small
    # raw-output headroom makes PPO's normal action-space clipping return an
    # exact deployed action of 3.0.  This is offline training only, not a
    # runtime safety filter.
    saturation_case = (teacher_action >= 2.999) | (repeated_required >= 2.999)
    safety_target = safety_target + float(raw_target_headroom) * saturation_case
    safety_target = np.clip(
        safety_target, -3.0, 3.0 + float(raw_target_headroom)
    ).astype(np.float32)
    return {
        "observations": observations,
        "targets": safety_target,
        "required_action": repeated_required.astype(np.float32),
        "selected_states": states,
        "selected_old_margin": old_margin,
    }


def train_repair(
    student_path: str,
    teacher_path: str,
    semantic_checkpoint: str,
    diagnostics_path: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    selection_margin: float,
    target_margin: float,
    raw_target_headroom: float,
    semantic_samples: int,
    local_weight: float,
    underbraking_weight: float,
    eval_episodes: int,
    device_name: str,
) -> Dict:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    teacher = load_controller(teacher_path)
    student = PPO.load(student_path, device=device_name)

    # Preserve v2 behavior on the same broad grid used during distillation.
    retention_observations = observation_grid(distance_scale, 128)
    retention_targets, _ = student.predict(
        retention_observations, deterministic=True
    )
    retention_targets = np.asarray(retention_targets, dtype=np.float32).reshape(-1, 1)
    local = local_repair_dataset(
        diagnostics_path,
        contract,
        distance_scale,
        teacher,
        selection_margin,
        semantic_samples,
        target_margin,
        raw_target_headroom,
    )

    observations = np.concatenate(
        (retention_observations, local["observations"]), axis=0
    )
    targets = np.concatenate((retention_targets, local["targets"]), axis=0)
    weights = np.concatenate(
        (
            np.ones(len(retention_observations), dtype=np.float32),
            np.full(len(local["observations"]), local_weight, dtype=np.float32),
        )
    )
    observation_tensor = torch.from_numpy(observations).to(student.device)
    target_tensor = torch.from_numpy(targets).to(student.device)
    weight_tensor = torch.from_numpy(weights).to(student.device)
    local_observation_tensor = torch.from_numpy(local["observations"]).to(
        student.device
    )
    local_required_tensor = torch.from_numpy(local["required_action"]).to(
        student.device
    )
    retention_observation_tensor = torch.from_numpy(retention_observations).to(
        student.device
    )
    retention_target_tensor = torch.from_numpy(retention_targets).to(student.device)

    actor_parameters = list(student.policy.mlp_extractor.policy_net.parameters())
    actor_parameters += list(student.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(student.policy.state_dict())
    best_score = (10**9, float("inf"), float("inf"))
    history = []
    started = time.time()
    student.policy.set_training_mode(True)

    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(observations), generator=generator)
        batch_losses = []
        for start in range(0, len(observations), batch_size):
            indices = permutation[start : start + batch_size].to(student.device)
            predicted = actor_output(student.policy, observation_tensor[indices])
            error = predicted - target_tensor[indices]
            underbraking = F.relu(target_tensor[indices] - predicted).square()
            per_sample = error.square() + underbraking_weight * underbraking
            loss = (
                per_sample.squeeze(1) * weight_tensor[indices]
            ).sum() / weight_tensor[indices].sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        with torch.no_grad():
            local_action = torch.clamp(
                actor_output(student.policy, local_observation_tensor), -3.0, 3.0
            )
            local_margin = local_action - local_required_tensor
            local_violation_count = int(torch.sum(local_margin < 0.0).cpu())
            minimum_local_margin = float(torch.min(local_margin).cpu())
            retention_action = torch.clamp(
                actor_output(student.policy, retention_observation_tensor),
                -3.0,
                3.0,
            )
            retention_mae = float(
                torch.mean(torch.abs(retention_action - retention_target_tensor)).cpu()
            )
        score = (
            local_violation_count,
            -minimum_local_margin,
            retention_mae,
        )
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(student.policy.state_dict())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = {
                "epoch": int(epoch),
                "loss": float(np.mean(batch_losses)),
                "local_violation_count": local_violation_count,
                "minimum_local_action_margin": minimum_local_margin,
                "retention_action_mae": retention_mae,
            }
            history.append(row)
            print(
                f"epoch={epoch} loss={row['loss']:.8f} "
                f"local_violations={local_violation_count} "
                f"min_local_margin={minimum_local_margin:.8f} "
                f"retention_mae={retention_mae:.8f}",
                flush=True,
            )

    student.policy.load_state_dict(best_state)
    student.policy.set_training_mode(False)
    checkpoint_path = output_dir / "standalone_ppo_repaired"
    student.save(checkpoint_path)

    repaired_retention_action, _ = student.predict(
        retention_observations, deterministic=True
    )
    retention_metrics = policy_action_metrics(
        repaired_retention_action, retention_targets
    )
    metrics = {
        "experiment": "local_certificate_margin_repair",
        "deployment_controller": "standalone_ppo_without_runtime_filter",
        "seed": int(seed),
        "source_student": student_path,
        "teacher": teacher_path,
        "diagnostics": diagnostics_path,
        "checkpoint": str(checkpoint_path) + ".zip",
        "selected_state_count": int(len(local["selected_states"])),
        "local_training_observation_count": int(len(local["observations"])),
        "retention_observation_count": int(len(retention_observations)),
        "selection_margin": float(selection_margin),
        "target_margin": float(target_margin),
        "raw_target_headroom": float(raw_target_headroom),
        "semantic_samples": int(semantic_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "local_weight": float(local_weight),
        "underbraking_weight": float(underbraking_weight),
        "training_seconds": float(time.time() - started),
        "history": history,
        "retention_action_match": retention_metrics,
        "student_evaluation": {},
    }
    for mode in MODES:
        print(f"evaluating repaired student: mode={mode}", flush=True)
        values = evaluate(
            student, contract, distance_scale, mode, eval_episodes, seed
        )
        metrics["student_evaluation"][mode] = values
        print(
            f"finished {mode}: success={100.0 * values['success_rate']:.2f}% "
            f"unsafe={100.0 * values['unsafe_rate']:.2f}% "
            f"timeout={100.0 * values['timeout_rate']:.2f}%",
            flush=True,
        )

    refinement = run_refinement(
        teacher_path,
        str(checkpoint_path) + ".zip",
        semantic_checkpoint,
        output_dir / "local_refinement",
        6.0,
        8.0,
        2.0,
        3.0,
        201,
        201,
        129,
        512,
        65536,
        1e-8,
    )
    metrics["local_refinement"] = {
        "status": refinement["status"],
        "student_violation_count": refinement["student"]["violation_count"],
        "minimum_next_margin_m": refinement["student"][
            "next_recoverability_margin_m"
        ]["minimum"],
        "minimum_action_surplus": refinement["student"][
            "action_surplus_over_required"
        ]["minimum"],
    }
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    print(f"\nresult directory: {output_dir.name}")
    print("[Repaired standalone PPO: no runtime safety filter]")
    print(
        f"selected states={metrics['selected_state_count']}, "
        f"local samples={metrics['local_training_observation_count']}, "
        f"retention MAE={retention_metrics['action_mae']:.6f}"
    )
    for mode in MODES:
        values = metrics["student_evaluation"][mode]
        print(
            f"{mode}: success={100.0 * values['success_rate']:.2f}%, "
            f"unsafe={100.0 * values['unsafe_rate']:.2f}%, "
            f"timeout={100.0 * values['timeout_rate']:.2f}%"
        )
    local_result = metrics["local_refinement"]
    print(
        f"local refinement: status={local_result['status']}, "
        f"violations={local_result['student_violation_count']}, "
        f"min_next_margin={local_result['minimum_next_margin_m']:.9f} m, "
        f"min_action_surplus={local_result['minimum_action_surplus']:.9f}"
    )
    print(f"checkpoint: {checkpoint_path}.zip")
    print(f"metrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--student",
        default="results/mvp/02_standalone_ppo_distilled_v2/standalone_ppo.zip",
    )
    parser.add_argument(
        "--teacher",
        default=(
            "results/mvp/02_safety_filter_anti_stall_compare/"
            "adaptive_anti_stall_filter.json"
        ),
    )
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--diagnostics",
        default=(
            "results/mvp/06_certificate_transfer_local_refinement/"
            "state_diagnostics.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/mvp/07_standalone_ppo_local_repair",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--selection-margin", type=float, default=0.1)
    parser.add_argument("--target-margin", type=float, default=0.02)
    parser.add_argument("--raw-target-headroom", type=float, default=0.0)
    parser.add_argument("--semantic-samples", type=int, default=65)
    parser.add_argument("--local-weight", type=float, default=10.0)
    parser.add_argument("--underbraking-weight", type=float, default=10.0)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train_repair(
        args.student,
        args.teacher,
        args.semantic_checkpoint,
        args.diagnostics,
        Path(args.output_dir),
        args.seed,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.selection_margin,
        args.target_margin,
        args.raw_target_headroom,
        args.semantic_samples,
        args.local_weight,
        args.underbraking_weight,
        args.eval_episodes,
        args.device,
    )


if __name__ == "__main__":
    main()
