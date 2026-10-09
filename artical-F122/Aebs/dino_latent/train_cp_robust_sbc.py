"""Train a scenario-based robust SBC for the frozen R2 PPO and CP contract.

The image path is represented by the calibrated physical next-state residual:

    q(distance), speed -> frozen PPO -> nominal next state
    image-path next state = nominal next state + w

where ``w`` is read from the trajectory-level conformal contract.  This first
implementation trains against a fixed residual grid and evaluates a denser
state/residual grid.  A zero sampled violation count is development evidence,
not a continuous-state formal certificate.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO

from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256


DT = 0.05
DOMAIN_DISTANCE_M = (5.0, 16.0)
DOMAIN_SPEED_MPS = (0.0, 3.0)
INITIAL_DISTANCE_M = (15.0, 16.0)
INITIAL_SPEED_MPS = (2.5, 3.0)
UNSAFE_DISTANCE_HIGH_M = 6.0
UNSAFE_SPEED_MPS = 0.5
ACTION_LIMIT = 3.0


class ScenarioBarrier(nn.Module):
    """Nonnegative neural barrier on physical distance and speed."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    @staticmethod
    def normalize(states):
        low = states.new_tensor([DOMAIN_DISTANCE_M[0], DOMAIN_SPEED_MPS[0]])
        span = states.new_tensor([
            DOMAIN_DISTANCE_M[1] - DOMAIN_DISTANCE_M[0],
            DOMAIN_SPEED_MPS[1] - DOMAIN_SPEED_MPS[0],
        ])
        return (states - low) / span

    def forward(self, states):
        return F.softplus(self.network(self.normalize(states))).squeeze(-1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_contract(path, representation_path, controller_path, calibration_h5):
    contract = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if contract.get("experiment") != "R2_physical_next_state_trajectory_conformal_contract_v1":
        errors.append("unexpected experiment")
    if contract.get("registered_before_calibration") is not True:
        errors.append("contract was not preregistered")
    if contract.get("representation_frozen") is not True or contract.get("controller_frozen") is not True:
        errors.append("representation/controller were not frozen")
    if contract.get("test_used") is not False:
        errors.append("test data was used")
    if contract.get("representation_sha256") != sha256(representation_path):
        errors.append("representation hash mismatch")
    if contract.get("controller_sha256") != sha256(controller_path):
        errors.append("controller hash mismatch")
    if not calibration_h5.is_file():
        errors.append("calibration H5 missing")
    elif contract.get("calibration_h5_sha256") != sha256(calibration_h5):
        errors.append("calibration H5 hash mismatch")
    box = contract.get("physical_residual_box", {})
    distance_box = box.get("distance_m")
    speed_box = box.get("speed_mps")
    if distance_box != [0.0, 0.0]:
        errors.append("first implementation requires zero distance residual")
    try:
        speed_low, speed_high = map(float, speed_box)
        if not (
            math.isfinite(speed_low) and math.isfinite(speed_high)
            and speed_low < 0.0 < speed_high
            and np.isclose(speed_low, -speed_high, atol=1e-9)
            and speed_high <= 2.0 * ACTION_LIMIT * DT + 1e-9
        ):
            errors.append("invalid or nonsymmetric speed residual box")
    except (TypeError, ValueError):
        speed_low = speed_high = float("nan")
        errors.append("missing speed residual box")
    conformal = contract.get("conformal", {})
    if not np.isclose(float(conformal.get("q_hat", np.nan)), speed_high, atol=1e-9):
        errors.append("q_hat and speed residual box disagree")
    if int(conformal.get("sample_count", -1)) < 459:
        errors.append("fewer than 459 calibration trajectories")
    if errors:
        raise ValueError("invalid R2 transition contract: " + "; ".join(errors))
    return contract, speed_low, speed_high


def policy_actions(controller, q_model, distance_scale_m, states, batch_size=8192):
    states = np.asarray(states, dtype=np.float32)
    output = []
    for start in range(0, len(states), batch_size):
        batch = states[start:start + batch_size]
        with torch.no_grad():
            latent = q_model(
                torch.from_numpy(batch[:, :1]) / float(distance_scale_m)
            ).numpy()
        observation = np.concatenate((latent, batch[:, 1:2]), axis=1)
        action = controller.predict(observation, deterministic=True)[0].reshape(-1)
        output.append(np.clip(action, -ACTION_LIMIT, ACTION_LIMIT))
    return np.concatenate(output).astype(np.float32)


def successor_scenarios(states, actions, residual_values):
    """Return [state, residual scenario, (distance,speed)] physical successors."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    residual_values = np.asarray(residual_values, dtype=np.float32).reshape(-1)
    next_distance = states[:, 0] - states[:, 1] * DT
    nominal_speed = np.clip(states[:, 1] - actions * DT, 0.0, 3.0)
    speed = np.clip(
        nominal_speed[:, None] + residual_values[None, :],
        DOMAIN_SPEED_MPS[0], DOMAIN_SPEED_MPS[1],
    )
    distance = np.repeat(next_distance[:, None], len(residual_values), axis=1)
    return np.stack((distance, speed), axis=2).astype(np.float32)


def safe_terminal_mask(states):
    states = np.asarray(states)
    distance = states[..., 0]
    speed = states[..., 1]
    goal = (distance <= UNSAFE_DISTANCE_HIGH_M) & (speed <= UNSAFE_SPEED_MPS)
    stopped = (speed <= 1e-6) & (distance > UNSAFE_DISTANCE_HIGH_M)
    return goal | stopped


def unsafe_mask(states):
    states = np.asarray(states)
    return (
        (states[..., 0] <= UNSAFE_DISTANCE_HIGH_M)
        & (states[..., 1] > UNSAFE_SPEED_MPS)
    )


def sample_uniform(rng, count, distance_range, speed_range):
    distance = rng.uniform(distance_range[0], distance_range[1], count)
    speed = rng.uniform(speed_range[0], speed_range[1], count)
    return np.stack((distance, speed), axis=1).astype(np.float32)


def sample_operational(rng, count):
    selected = []
    remaining = count
    while remaining:
        candidates = sample_uniform(
            rng, max(remaining * 2, 1024), DOMAIN_DISTANCE_M, DOMAIN_SPEED_MPS
        )
        keep = ~(unsafe_mask(candidates) | safe_terminal_mask(candidates))
        values = candidates[keep][:remaining]
        selected.append(values)
        remaining -= len(values)
    return np.concatenate(selected, axis=0)


def grid_states(distance_range, speed_range, size):
    distance = np.linspace(distance_range[0], distance_range[1], size, dtype=np.float32)
    speed = np.linspace(speed_range[0], speed_range[1], size, dtype=np.float32)
    mesh_d, mesh_v = np.meshgrid(distance, speed)
    return np.stack((mesh_d.reshape(-1), mesh_v.reshape(-1)), axis=1)


def barrier_values(barrier, states, device, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            values.append(barrier(
                torch.from_numpy(states[start:start + batch_size]).to(device)
            ).cpu().numpy())
    return np.concatenate(values)


def robust_next_values(barrier, successors, device, batch_size):
    flat = successors.reshape(-1, 2)
    values = barrier_values(barrier, flat, device, batch_size).reshape(successors.shape[:2])
    terminal = safe_terminal_mask(successors)
    values[terminal] = 0.0
    return values.max(axis=1)


def evaluate(
    barrier, controller, q_model, distance_scale_m, speed_low, speed_high,
    grid_size, residual_samples, batch_size, init_target, unsafe_target, device,
):
    all_states = grid_states(DOMAIN_DISTANCE_M, DOMAIN_SPEED_MPS, grid_size)
    operational = all_states[~(unsafe_mask(all_states) | safe_terminal_mask(all_states))]
    actions = policy_actions(controller, q_model, distance_scale_m, operational, batch_size)
    residuals = np.linspace(speed_low, speed_high, residual_samples, dtype=np.float32)
    successors = successor_scenarios(operational, actions, residuals)
    current = barrier_values(barrier, operational, device, batch_size)
    robust_next = robust_next_values(barrier, successors, device, batch_size)
    margins = current - robust_next

    initial = grid_states(INITIAL_DISTANCE_M, INITIAL_SPEED_MPS, grid_size)
    # Include d just below 5 because a one-step successor from the declared
    # domain can cross the lower distance edge before terminal classification.
    unsafe = grid_states((4.8, UNSAFE_DISTANCE_HIGH_M),
                         (UNSAFE_SPEED_MPS + 1e-4, 3.0), grid_size)
    initial_values = barrier_values(barrier, initial, device, batch_size)
    unsafe_values = barrier_values(barrier, unsafe, device, batch_size)
    decrease_violations = margins < -1e-6
    init_violations = initial_values > init_target + 1e-6
    unsafe_violations = unsafe_values < unsafe_target - 1e-6
    worst_index = int(np.argmin(margins))
    status = "sampled_grid_pass_ready_for_formal_bounds" if (
        not decrease_violations.any()
        and not init_violations.any()
        and not unsafe_violations.any()
    ) else "sampled_grid_violated"
    return {
        "status": status,
        "claim_level": "sampled state/residual grid only; not a continuous formal certificate",
        "grid_size": int(grid_size),
        "residual_samples": int(residual_samples),
        "operational_states": int(len(operational)),
        "decrease_violation_count": int(decrease_violations.sum()),
        "min_decrease_margin": float(margins.min()),
        "mean_decrease_margin": float(margins.mean()),
        "initial_violation_count": int(init_violations.sum()),
        "initial_max": float(initial_values.max()),
        "initial_target_max": float(init_target),
        "unsafe_violation_count": int(unsafe_violations.sum()),
        "unsafe_min": float(unsafe_values.min()),
        "unsafe_target_min": float(unsafe_target),
        "worst_case": {
            "state": operational[worst_index].astype(float).tolist(),
            "current_barrier": float(current[worst_index]),
            "robust_next_barrier": float(robust_next[worst_index]),
            "margin": float(margins[worst_index]),
        },
    }


def train(args):
    set_seed(args.seed)
    controller_path = args.ppo_dir / "latent_ppo.zip"
    controller_metrics_path = args.ppo_dir / "metrics.json"
    required = [
        args.contract, args.calibration_h5, args.representation,
        controller_path, controller_metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing inputs: " + ", ".join(missing))
    if args.output_dir.exists():
        raise FileExistsError("output directory exists: %s" % args.output_dir)

    contract, speed_low, speed_high = load_contract(
        args.contract, args.representation, controller_path, args.calibration_h5
    )
    representation = torch.load(args.representation, map_location="cpu")
    controller_metrics = json.loads(controller_metrics_path.read_text(encoding="utf-8"))
    if (
        representation.get("variant") != "R2_group_alignment"
        or controller_metrics.get("representation_sha256") != sha256(args.representation)
        or controller_metrics.get("checkpoint_sha256") != sha256(controller_path)
        or controller_metrics.get("test_used") is not False
    ):
        raise ValueError("R2/PPO provenance mismatch")

    q_model = PhysicalToLatent().eval()
    q_model.load_state_dict(representation["physical_to_latent_state_dict"])
    for parameter in q_model.parameters():
        parameter.requires_grad_(False)
    distance_scale_m = float(representation["distance_scale_m"])
    controller = PPO.load(controller_path, device="cpu")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    barrier = ScenarioBarrier().to(device)
    optimizer = torch.optim.Adam(barrier.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    operational = sample_operational(rng, args.train_states)
    initial = sample_uniform(
        rng, args.region_samples, INITIAL_DISTANCE_M, INITIAL_SPEED_MPS
    )
    unsafe = sample_uniform(
        rng, args.region_samples, (4.8, UNSAFE_DISTANCE_HIGH_M),
        (UNSAFE_SPEED_MPS + 1e-4, 3.0),
    )
    terminal = np.concatenate((
        sample_uniform(rng, args.region_samples // 2,
                       (5.0, UNSAFE_DISTANCE_HIGH_M), (0.0, UNSAFE_SPEED_MPS)),
        sample_uniform(rng, args.region_samples - args.region_samples // 2,
                       (UNSAFE_DISTANCE_HIGH_M, 16.0), (0.0, 1e-6)),
    ), axis=0)
    actions = policy_actions(controller, q_model, distance_scale_m, operational)
    train_residuals = np.linspace(
        speed_low, speed_high, args.train_residual_samples, dtype=np.float32
    )
    successors = successor_scenarios(operational, actions, train_residuals)

    args.output_dir.mkdir(parents=True)
    config = {
        "experiment": "R2_CP_contract_scenario_robust_SBC_v1",
        "claim_level": "scenario training and sampled-grid verification only",
        "seed": args.seed,
        "contract": str(args.contract),
        "contract_sha256": sha256(args.contract),
        "calibration_h5": str(args.calibration_h5),
        "calibration_h5_sha256": sha256(args.calibration_h5),
        "representation": str(args.representation),
        "representation_sha256": sha256(args.representation),
        "controller": str(controller_path),
        "controller_sha256": sha256(controller_path),
        "conformal": contract["conformal"],
        "physical_residual_box": contract["physical_residual_box"],
        "training": {
            "epochs": args.epochs,
            "train_states": args.train_states,
            "region_samples": args.region_samples,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_residual_samples": args.train_residual_samples,
            "init_target": args.init_target,
            "unsafe_target": args.unsafe_target,
            "decrease_weight": args.decrease_weight,
            "region_weight": args.region_weight,
            "terminal_weight": args.terminal_weight,
            "topk_fraction": args.topk_fraction,
        },
        "test_used": False,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "training_scenarios.npz",
        operational=operational,
        initial=initial,
        unsafe=unsafe,
        terminal=terminal,
        residuals=train_residuals,
    )

    history = []
    started = time.time()
    order = np.arange(len(operational))
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        totals = []
        decreases = []
        init_losses = []
        unsafe_losses = []
        terminal_losses = []
        violation_rates = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch_states = torch.from_numpy(operational[indices]).to(device)
            batch_successors_np = successors[indices]
            batch_successors = torch.from_numpy(batch_successors_np).to(device)
            batch_init = torch.from_numpy(
                initial[rng.integers(0, len(initial), len(indices))]
            ).to(device)
            batch_unsafe = torch.from_numpy(
                unsafe[rng.integers(0, len(unsafe), len(indices))]
            ).to(device)
            batch_terminal = torch.from_numpy(
                terminal[rng.integers(0, len(terminal), len(indices))]
            ).to(device)

            current = barrier(batch_states)
            next_values = barrier(batch_successors.reshape(-1, 2)).reshape(
                len(indices), args.train_residual_samples
            )
            terminal_mask = torch.from_numpy(
                safe_terminal_mask(batch_successors_np)
            ).to(device)
            next_values = torch.where(terminal_mask, torch.zeros_like(next_values), next_values)
            robust_next = next_values.max(dim=1).values
            per_decrease = F.relu(robust_next - current)
            topk_count = max(1, int(math.ceil(len(per_decrease) * args.topk_fraction)))
            decrease_loss = per_decrease.square().mean() + torch.topk(
                per_decrease.square(), topk_count
            ).values.mean()
            init_loss = F.relu(barrier(batch_init) - args.init_target).square().mean()
            unsafe_loss = F.relu(args.unsafe_target - barrier(batch_unsafe)).square().mean()
            terminal_loss = barrier(batch_terminal).square().mean()
            loss = (
                args.decrease_weight * decrease_loss
                + args.region_weight * (init_loss + unsafe_loss)
                + args.terminal_weight * terminal_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(barrier.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            decreases.append(float(decrease_loss.detach().cpu()))
            init_losses.append(float(init_loss.detach().cpu()))
            unsafe_losses.append(float(unsafe_loss.detach().cpu()))
            terminal_losses.append(float(terminal_loss.detach().cpu()))
            violation_rates.append(float((per_decrease > 1e-6).float().mean().detach().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            record = {
                "epoch": epoch,
                "loss": float(np.mean(totals)),
                "decrease_loss": float(np.mean(decreases)),
                "init_loss": float(np.mean(init_losses)),
                "unsafe_loss": float(np.mean(unsafe_losses)),
                "terminal_loss": float(np.mean(terminal_losses)),
                "train_decrease_violation_rate": float(np.mean(violation_rates)),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    checkpoint = {
        "experiment": config["experiment"],
        "state_dict": barrier.state_dict(),
        "contract_sha256": config["contract_sha256"],
        "representation_sha256": config["representation_sha256"],
        "controller_sha256": config["controller_sha256"],
        "input_units": ["distance_m", "speed_mps"],
        "normalization": {"distance_m": [5.0, 16.0], "speed_mps": [0.0, 3.0]},
    }
    torch.save(checkpoint, args.output_dir / "barrier.pt")
    barrier.eval()
    verification = evaluate(
        barrier, controller, q_model, distance_scale_m, speed_low, speed_high,
        args.grid_size, args.verify_residual_samples, args.batch_size,
        args.init_target, args.unsafe_target, device,
    )
    metrics = {
        **config,
        "runtime_seconds": float(time.time() - started),
        "history": history,
        "verification": verification,
        "checkpoint": str(args.output_dir / "barrier.pt"),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("\n[R2 CP-contract scenario robust SBC]")
    print("q_hat=%.9f residual=[%.9f, %.9f]" % (
        float(contract["conformal"]["q_hat"]), speed_low, speed_high
    ))
    print("status:", verification["status"])
    print("decrease violations: %d/%d min_margin=%.9f" % (
        verification["decrease_violation_count"],
        verification["operational_states"],
        verification["min_decrease_margin"],
    ))
    print("initial violations: %d max=%.9f target<=%.9f" % (
        verification["initial_violation_count"], verification["initial_max"],
        verification["initial_target_max"],
    ))
    print("unsafe violations: %d min=%.9f target>=%.9f" % (
        verification["unsafe_violation_count"], verification["unsafe_min"],
        verification["unsafe_target_min"],
    ))
    print("worst case:", verification["worst_case"])
    print("metrics:", args.output_dir / "metrics.json")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract", type=Path,
        default=Path("results/dino_expanded_v1/14_transition_contract_spvc_v1/contract.json"),
    )
    parser.add_argument(
        "--calibration-h5", type=Path,
        default=Path("results/dino_expanded_v1/13_spvc_calibration_trajectories_v1/trajectories.h5"),
    )
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"),
    )
    parser.add_argument(
        "--ppo-dir", type=Path,
        default=Path("results/dino_expanded_v1/09_R2_ppo"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/15_cp_robust_sbc_v1"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train-states", type=int, default=20000)
    parser.add_argument("--region-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-residual-samples", type=int, default=5)
    parser.add_argument("--verify-residual-samples", type=int, default=33)
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--init-target", type=float, default=0.01)
    parser.add_argument("--unsafe-target", type=float, default=1.0)
    parser.add_argument("--decrease-weight", type=float, default=100.0)
    parser.add_argument("--region-weight", type=float, default=20.0)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--topk-fraction", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    if (
        min(args.epochs, args.train_states, args.region_samples, args.batch_size,
            args.train_residual_samples, args.verify_residual_samples,
            args.grid_size, args.log_every) < 1
        or not 0.0 < args.topk_fraction <= 1.0
        or args.learning_rate <= 0.0
        or args.init_target < 0.0
        or args.unsafe_target <= args.init_target
    ):
        parser.error("invalid training or verification settings")
    train(args)


if __name__ == "__main__":
    main()
