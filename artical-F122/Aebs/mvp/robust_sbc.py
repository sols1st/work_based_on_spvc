"""Minimal robust SBC trainer and fixed-grid verifier for AEBS MVP."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO

from Aebs.VT.utils import MLP
from Aebs.semantic.robust_controller import load_contract
from Aebs.uncertainty.model import StateDependentUncertainty


DISTURBANCE_LABELS = (
    "mean",
    "support_low",
    "support_high",
    "low_distance_high_speed",
    "high_distance_low_speed",
)
SEMANTIC_MULTIPLIERS = (-1.0, 0.0, 1.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def controller_actions(controller: PPO, states: np.ndarray) -> np.ndarray:
    actions, _ = controller.predict(states.astype(np.float32), deterministic=True)
    return np.asarray(actions, dtype=np.float32).reshape(-1)


def next_state(states: np.ndarray, actions: np.ndarray, distance_scale: float) -> np.ndarray:
    distance_m = states[:, 0] * distance_scale
    speed = states[:, 1]
    next_distance_m = distance_m - speed * 0.05
    next_speed = np.clip(speed - actions * 0.05, 0.0, 3.0)
    return np.stack((next_distance_m / distance_scale, next_speed), axis=1)


def sample_box(low: Tuple[float, float], high: Tuple[float, float], count: int, rng: np.random.Generator) -> np.ndarray:
    low_array = np.asarray(low, dtype=np.float64)
    high_array = np.asarray(high, dtype=np.float64)
    return rng.uniform(low_array, high_array, size=(count, 2)).astype(np.float32)


def verifier_grid_states(distance_scale: float, grid_size: int) -> np.ndarray:
    distances = np.linspace(5.0 / distance_scale, 16.0 / distance_scale, grid_size)
    speeds = np.linspace(0.0, 3.0, grid_size)
    mesh_d, mesh_v = np.meshgrid(distances, speeds)
    return np.stack((mesh_d.reshape(-1), mesh_v.reshape(-1)), axis=1).astype(np.float32)


def robust_successors(
    states: np.ndarray,
    controller: PPO,
    uncertainty: StateDependentUncertainty,
    distance_scale: float,
    semantic_radii: np.ndarray,
) -> np.ndarray:
    candidates = []
    for semantic_multiplier in SEMANTIC_MULTIPLIERS:
        semantic_states = states.copy()
        semantic_states[:, 0] = semantic_states[:, 0] + semantic_multiplier * semantic_radii
        semantic_states[:, 0] = np.clip(semantic_states[:, 0], 5.0 / distance_scale, 16.0 / distance_scale)
        actions = controller_actions(controller, semantic_states)
        base_next = next_state(states, actions, distance_scale)
        box = uncertainty.lookup(states, distance_scale)
        disturbance_points = [
            box["mean"],
            box["support_low"],
            box["support_high"],
            np.stack((box["support_low"][:, 0], box["support_high"][:, 1]), axis=1),
            np.stack((box["support_high"][:, 0], box["support_low"][:, 1]), axis=1),
        ]
        for disturbance in disturbance_points:
            perturbed = base_next + disturbance
            perturbed[:, 0] = np.clip(perturbed[:, 0], 5.0 / distance_scale, 16.0 / distance_scale)
            perturbed[:, 1] = np.clip(perturbed[:, 1], 0.0, 3.0)
            candidates.append(perturbed)
    return np.stack(candidates, axis=1).astype(np.float32)


def candidate_label(index: int) -> Dict[str, object]:
    semantic_index = index // len(DISTURBANCE_LABELS)
    disturbance_index = index % len(DISTURBANCE_LABELS)
    return {
        "semantic_multiplier": SEMANTIC_MULTIPLIERS[semantic_index],
        "disturbance": DISTURBANCE_LABELS[disturbance_index],
    }


def evaluate_grid(
    barrier: MLP,
    controller: PPO,
    uncertainty: StateDependentUncertainty,
    semantic_checkpoint: str,
    data_path: str,
    grid_size: int,
    batch_size: int,
    epsilon: float,
    device: torch.device,
) -> Dict:
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        distance_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, distance_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")
    states = verifier_grid_states(distance_scale, grid_size)
    margins = []
    worst = {
        "margin": float("inf"),
        "state": None,
    }
    barrier.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            distance_m = batch[:, 0] * distance_scale
            radii = contract.radius(distance_m).astype(np.float32)
            successors = robust_successors(batch, controller, uncertainty, distance_scale, radii)
            current = barrier(torch.from_numpy(batch).to(device)).view(-1)
            next_values = barrier(torch.from_numpy(successors.reshape(-1, 2)).to(device)).view(len(batch), -1)
            robust_next, robust_index = next_values.max(dim=1)
            margin = (current - robust_next - epsilon).detach().cpu().numpy()
            margins.append(margin)
            local_index = int(np.argmin(margin))
            if float(margin[local_index]) < worst["margin"]:
                label = candidate_label(int(robust_index[local_index].detach().cpu()))
                worst = {
                    "margin": float(margin[local_index]),
                    "state": batch[local_index].astype(float).tolist(),
                    "state_units": ["normalized_distance", "m_per_s"],
                    "distance_m": float(batch[local_index, 0] * distance_scale),
                    "speed": float(batch[local_index, 1]),
                    "current_barrier": float(current[local_index].detach().cpu()),
                    "robust_next_barrier": float(robust_next[local_index].detach().cpu()),
                    "semantic_multiplier": label["semantic_multiplier"],
                    "disturbance": label["disturbance"],
                    "successor": successors[local_index, int(robust_index[local_index].detach().cpu())].astype(float).tolist(),
                }
    margins = np.concatenate(margins)
    violation_count = int(np.sum(margins < 0.0))
    status = "verified" if violation_count == 0 else "violated"
    return {
        "status": status,
        "grid_size": int(grid_size),
        "checked_states": int(len(states)),
        "violation_count": violation_count,
        "min_margin": float(np.min(margins)),
        "mean_margin": float(np.mean(margins)),
        "worst_case": worst,
    }


def train_barrier(
    semantic_checkpoint: str,
    uncertainty_path: str,
    controller_path: str,
    data_path: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    train_states: int,
    batch_size: int,
    learning_rate: float,
    epsilon: float,
    grid_size: int,
    decrease_weight: float,
    init_weight: float,
    unsafe_weight: float,
    square_output: bool,
    region_warmup_epochs: int,
    warmup_decrease_weight: float,
    init_target: float,
    unsafe_target: float,
    include_verifier_grid_train: bool,
) -> Dict:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, checkpoint_scale = load_contract(semantic_checkpoint)
    with h5py.File(data_path, "r") as stream:
        distance_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, distance_scale):
        raise ValueError("semantic checkpoint and dataset use different distance scaling")
    controller = PPO.load(controller_path, device="cpu")
    uncertainty = StateDependentUncertainty.from_npz(uncertainty_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=square_output).to(device)
    optimizer = torch.optim.Adam(barrier.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    random_states = sample_box((5.0 / distance_scale, 0.0), (16.0 / distance_scale, 3.0), train_states, rng)
    if include_verifier_grid_train:
        states = np.concatenate((random_states, verifier_grid_states(distance_scale, grid_size)), axis=0)
    else:
        states = random_states
    init_states = sample_box((15.0 / distance_scale, 2.5), (16.0 / distance_scale, 3.0), max(512, batch_size), rng)
    unsafe_states = sample_box((5.0 / distance_scale, 0.5), (6.0 / distance_scale, 3.0), max(512, batch_size), rng)
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(states))
        epoch_losses = []
        epoch_decrease_losses = []
        epoch_init_losses = []
        epoch_unsafe_losses = []
        epoch_violations = []
        active_decrease_weight = warmup_decrease_weight if epoch <= region_warmup_epochs else decrease_weight
        for start in range(0, len(states), batch_size):
            batch = states[permutation[start : start + batch_size]]
            distance_m = batch[:, 0] * distance_scale
            radii = contract.radius(distance_m).astype(np.float32)
            successors = robust_successors(batch, controller, uncertainty, distance_scale, radii)
            batch_tensor = torch.from_numpy(batch).to(device)
            successor_tensor = torch.from_numpy(successors.reshape(-1, 2)).to(device)
            current = barrier(batch_tensor).view(-1)
            next_values = barrier(successor_tensor).view(len(batch), -1)
            robust_next = next_values.max(dim=1).values
            decrease_loss = F.relu(robust_next - current + epsilon).mean()
            init_value = barrier(torch.from_numpy(init_states).to(device)).view(-1)
            unsafe_value = barrier(torch.from_numpy(unsafe_states).to(device)).view(-1)
            init_loss = F.relu(init_value - init_target).mean()
            unsafe_loss = F.relu(unsafe_target - unsafe_value).mean()
            loss = active_decrease_weight * decrease_loss + init_weight * init_loss + unsafe_weight * unsafe_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(barrier.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_decrease_losses.append(float(decrease_loss.detach().cpu()))
            epoch_init_losses.append(float(init_loss.detach().cpu()))
            epoch_unsafe_losses.append(float(unsafe_loss.detach().cpu()))
            epoch_violations.append(float(torch.mean((robust_next >= current).float()).detach().cpu()))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            record = {
                "epoch": int(epoch),
                "loss": float(np.mean(epoch_losses)),
                "decrease_loss": float(np.mean(epoch_decrease_losses)),
                "init_loss": float(np.mean(epoch_init_losses)),
                "unsafe_loss": float(np.mean(epoch_unsafe_losses)),
                "active_decrease_weight": float(active_decrease_weight),
                "train_robust_decrease_violation_rate": float(np.mean(epoch_violations)),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    torch.save(barrier.state_dict(), output_dir / "barrier.pt")
    verification = evaluate_grid(
        barrier,
        controller,
        uncertainty,
        semantic_checkpoint,
        data_path,
        grid_size,
        batch_size,
        epsilon,
        device,
    )
    metrics = {
        "experiment": "robust_sbc_mvp",
        "seed": seed,
        "controller": controller_path,
        "epochs": epochs,
        "train_states": train_states,
        "effective_train_states": int(len(states)),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epsilon": epsilon,
        "decrease_weight": decrease_weight,
        "init_weight": init_weight,
        "unsafe_weight": unsafe_weight,
        "square_output": bool(square_output),
        "region_warmup_epochs": int(region_warmup_epochs),
        "warmup_decrease_weight": float(warmup_decrease_weight),
        "init_target": float(init_target),
        "unsafe_target": float(unsafe_target),
        "include_verifier_grid_train": bool(include_verifier_grid_train),
        "runtime_seconds": float(time.time() - started),
        "history": history,
        "verification": verification,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--uncertainty", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", default="results/mvp/04_robust_sbc")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--train-states", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--grid-size", type=int, default=80)
    parser.add_argument("--decrease-weight", type=float, default=100.0)
    parser.add_argument("--init-weight", type=float, default=1.0)
    parser.add_argument("--unsafe-weight", type=float, default=1.0)
    parser.add_argument("--square-output", action="store_true")
    parser.add_argument("--region-warmup-epochs", type=int, default=25)
    parser.add_argument("--warmup-decrease-weight", type=float, default=0.0)
    parser.add_argument("--init-target", type=float, default=1.0)
    parser.add_argument("--unsafe-target", type=float, default=10.0)
    parser.add_argument("--include-verifier-grid-train", action="store_true")
    args = parser.parse_args()
    metrics = train_barrier(
        args.semantic_checkpoint,
        args.uncertainty,
        args.controller,
        args.data,
        Path(args.output_dir),
        args.seed,
        args.epochs,
        args.train_states,
        args.batch_size,
        args.learning_rate,
        args.epsilon,
        args.grid_size,
        args.decrease_weight,
        args.init_weight,
        args.unsafe_weight,
        args.square_output,
        args.region_warmup_epochs,
        args.warmup_decrease_weight,
        args.init_target,
        args.unsafe_target,
        args.include_verifier_grid_train,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
