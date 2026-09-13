"""Distill the successful AEBS safety-filter policy into a standalone PPO.

The safety filter is used only as an offline teacher.  The saved student and
all student evaluations call PPO directly, with no runtime safety filter.
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

from Aebs.semantic.robust_controller import evaluate, load_contract
from Aebs.semantic.safety_filter import load_controller


MODES = ("exact", "uniform", "random_boundary", "worst_endpoint")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def observation_grid(distance_scale: float, grid_size: int) -> np.ndarray:
    distance = np.linspace(5.0 / distance_scale, 16.0 / distance_scale, grid_size)
    speed = np.linspace(0.0, 3.0, grid_size)
    mesh_distance, mesh_speed = np.meshgrid(distance, speed)
    return np.stack(
        (mesh_distance.reshape(-1), mesh_speed.reshape(-1)), axis=1
    ).astype(np.float32)


def actor_output(policy, observations: torch.Tensor) -> torch.Tensor:
    latent = policy.mlp_extractor.policy_net(observations)
    return policy.action_net(latent)


def imitation_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    speed: torch.Tensor,
    underbraking_weight: float,
    low_speed_weight: float,
) -> torch.Tensor:
    squared_error = (predicted - target).square()
    underbraking = F.relu(target - predicted).square()
    sample_weight = torch.where(
        speed <= 0.6,
        torch.full_like(speed, low_speed_weight),
        torch.ones_like(speed),
    )
    return (
        sample_weight
        * (squared_error.squeeze(1) + underbraking_weight * underbraking.squeeze(1))
    ).mean()


def policy_action_metrics(predicted: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = np.asarray(predicted).reshape(-1) - np.asarray(target).reshape(-1)
    return {
        "action_mae": float(np.mean(np.abs(error))),
        "action_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "action_max_abs_error": float(np.max(np.abs(error))),
        "underbraking_rate": float(np.mean(error < -0.05)),
        "overbraking_rate": float(np.mean(error > 0.05)),
    }


def train_standalone_ppo(
    baseline_path: str,
    teacher_path: str,
    semantic_checkpoint: str,
    output_dir: Path,
    seed: int,
    grid_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    underbraking_weight: float,
    low_speed_weight: float,
    eval_episodes: int,
    device_name: str,
) -> Dict:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(semantic_checkpoint)
    teacher = load_controller(teacher_path)
    baseline = PPO.load(baseline_path, device="cpu")
    student = PPO.load(baseline_path, device=device_name)

    observations = observation_grid(distance_scale, grid_size)
    teacher_actions, _ = teacher.predict(observations, deterministic=True)
    teacher_actions = np.asarray(teacher_actions, dtype=np.float32).reshape(-1, 1)
    observation_tensor = torch.from_numpy(observations).to(student.device)
    target_tensor = torch.from_numpy(teacher_actions).to(student.device)
    speed_tensor = observation_tensor[:, 1]

    actor_parameters = list(student.policy.mlp_extractor.policy_net.parameters())
    actor_parameters += list(student.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = float("inf")
    best_state = None
    history = []
    started = time.time()
    student.policy.set_training_mode(True)
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(observations), generator=generator)
        batch_losses = []
        for start in range(0, len(observations), batch_size):
            batch_indices = permutation[start : start + batch_size].to(student.device)
            predicted = actor_output(student.policy, observation_tensor[batch_indices])
            loss = imitation_loss(
                predicted,
                target_tensor[batch_indices],
                speed_tensor[batch_indices],
                underbraking_weight,
                low_speed_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_loss = float(np.mean(batch_losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(student.policy.state_dict())
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            history.append({"epoch": epoch, "loss": epoch_loss})
            print(f"epoch={epoch} imitation_loss={epoch_loss:.8f}", flush=True)

    if best_state is None:
        raise RuntimeError("standalone PPO training produced no checkpoint")
    student.policy.load_state_dict(best_state)
    student.policy.set_training_mode(False)
    checkpoint_path = output_dir / "standalone_ppo"
    student.save(checkpoint_path)

    student_actions, _ = student.predict(observations, deterministic=True)
    action_metrics = policy_action_metrics(student_actions, teacher_actions)
    metrics = {
        "experiment": "safety_filter_teacher_to_standalone_ppo",
        "deployment_controller": "standalone_ppo_without_safety_filter",
        "seed": int(seed),
        "teacher": teacher_path,
        "baseline": baseline_path,
        "checkpoint": str(checkpoint_path) + ".zip",
        "grid_size": int(grid_size),
        "training_samples": int(len(observations)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "underbraking_weight": float(underbraking_weight),
        "low_speed_weight": float(low_speed_weight),
        "best_imitation_loss": float(best_loss),
        "training_seconds": float(time.time() - started),
        "history": history,
        "action_match": action_metrics,
        "baseline_evaluation": {},
        "teacher_evaluation": {},
        "student_evaluation": {},
    }
    for mode in MODES:
        metrics["baseline_evaluation"][mode] = evaluate(
            baseline, contract, distance_scale, mode, eval_episodes, seed
        )
        metrics["teacher_evaluation"][mode] = evaluate(
            teacher, contract, distance_scale, mode, eval_episodes, seed
        )
        # This call is deliberately made on the PPO object, not the teacher.
        metrics["student_evaluation"][mode] = evaluate(
            student, contract, distance_scale, mode, eval_episodes, seed
        )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)

    print("\n[Standalone PPO distilled from safety-filter teacher]")
    print(
        f"action MAE={action_metrics['action_mae']:.6f}, "
        f"max error={action_metrics['action_max_abs_error']:.6f}"
    )
    for mode in MODES:
        values = metrics["student_evaluation"][mode]
        print(
            f"{mode}: success={100.0 * values['success_rate']:.2f}%, "
            f"unsafe={100.0 * values['unsafe_rate']:.2f}%, "
            f"timeout={100.0 * values['timeout_rate']:.2f}%, "
            f"steps={values['mean_steps']:.1f}"
        )
    print(f"checkpoint: {checkpoint_path}.zip")
    print(f"metrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="Aebs/controller/best_model/best_model.zip",
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
        "--output-dir",
        default="results/mvp/02_standalone_ppo_distilled",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--underbraking-weight", type=float, default=4.0)
    parser.add_argument("--low-speed-weight", type=float, default=4.0)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train_standalone_ppo(
        args.baseline,
        args.teacher,
        args.semantic_checkpoint,
        Path(args.output_dir),
        args.seed,
        args.grid_size,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.underbraking_weight,
        args.low_speed_weight,
        args.eval_episodes,
        args.device,
    )


if __name__ == "__main__":
    main()
