"""Fine-tune the semantic controller with certified perception-error augmentation."""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.semantic.conformal import StateConditionalErrorContract
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import (
    EPISODE_OUTCOMES,
    SUCCESS,
    TIMEOUT,
    UNSAFE,
    classify_terminal_outcome,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_contract(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    values = checkpoint["state_contract"]
    contract = StateConditionalErrorContract(
        alpha=float(values["alpha"]),
        bin_edges=tuple(values["bin_edges"]),
        radii=tuple(values["radii"]),
        calibration_sizes=tuple(values["calibration_sizes"]),
    )
    return contract, float(checkpoint["distance_scale_m"])


class PerceptionNoiseEnv(AebsEnv):
    """AEBS dynamics with exact velocity and a noisy semantic distance observation."""

    def __init__(self, distance_scale: float, contract: StateConditionalErrorContract, seed: int):
        super().__init__(distance_scale)
        self.contract = contract
        self.rng = np.random.default_rng(seed)

    def _observe(self, true_observation: np.ndarray) -> np.ndarray:
        observation = true_observation.copy()
        distance_m = observation[0] * self.std1
        radius = float(self.contract.radius(np.array([distance_m]))[0])
        draw = self.rng.random()
        if draw < 0.50:
            error = 0.0
        elif draw < 0.75:
            error = self.rng.uniform(-radius, radius)
        else:
            error = radius * (-1.0 if self.rng.random() < 0.5 else 1.0)
        observation[0] = np.clip(
            observation[0] + error,
            self.observation_space.low[0],
            self.observation_space.high[0],
        )
        return observation.astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            np.random.seed(seed)
        observation, info = super().reset(seed=seed, options=options)
        return self._observe(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return self._observe(observation), reward, terminated, truncated, info


def action(model: PPO, distance_norm: float, speed: float) -> float:
    prediction, _ = model.predict(
        np.array([distance_norm, speed], dtype=np.float32), deterministic=True
    )
    return float(np.clip(np.asarray(prediction).reshape(-1)[0], -3.0, 3.0))


def action_with_diagnostics(model, distance_norm: float, speed: float):
    observation = np.array([distance_norm, speed], dtype=np.float32)
    if hasattr(model, "predict_with_diagnostics"):
        prediction, diagnostics = model.predict_with_diagnostics(observation, deterministic=True)
        scalar_diagnostics = {
            key: float(np.asarray(value).reshape(-1)[0])
            for key, value in diagnostics.items()
        }
    else:
        prediction, _ = model.predict(observation, deterministic=True)
        scalar_diagnostics = {}
    chosen_action = float(np.clip(np.asarray(prediction).reshape(-1)[0], -3.0, 3.0))
    return chosen_action, scalar_diagnostics


def evaluate(
    model: PPO,
    contract: StateConditionalErrorContract,
    distance_scale: float,
    mode: str,
    episodes: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    outcome_counts = {name: 0 for name in EPISODE_OUTCOMES}
    returns, lengths, minimum_distances, success_lengths = [], [], [], []
    intervention_steps = 0
    filter_steps = 0
    extra_braking_sum = 0.0
    used_radius_sum = 0.0
    overshoot_cap_steps = 0
    for _ in range(episodes):
        distance_m = float(rng.uniform(15.0, 16.0))
        speed = float(rng.uniform(2.5, 3.0))
        episode_return = 0.0
        minimum_distance = distance_m
        outcome = TIMEOUT
        for step in range(400):
            distance_norm = distance_m / distance_scale
            radius = float(contract.radius(np.array([distance_m]))[0])
            if mode == "exact":
                observed_distance_norm = distance_norm
            elif mode == "uniform":
                observed_distance_norm = distance_norm + rng.uniform(-radius, radius)
            elif mode == "random_boundary":
                observed_distance_norm = distance_norm + radius * rng.choice((-1.0, 1.0))
            elif mode == "worst_endpoint":
                # Positive action is braking in these dynamics; the smaller endpoint action is safety-worst.
                endpoint_actions = []
                endpoint_diagnostics = []
                for endpoint in (distance_norm - radius, distance_norm + radius):
                    clipped_endpoint = float(
                        np.clip(endpoint, 5.0 / distance_scale, 16.0 / distance_scale)
                    )
                    candidate_action, candidate_diagnostics = action_with_diagnostics(
                        model, clipped_endpoint, speed
                    )
                    endpoint_actions.append(candidate_action)
                    endpoint_diagnostics.append(candidate_diagnostics)
                worst_index = int(np.argmin(endpoint_actions))
                chosen_action = endpoint_actions[worst_index]
                diagnostics = endpoint_diagnostics[worst_index]
            else:
                raise ValueError(f"unknown evaluation mode: {mode}")

            if mode != "worst_endpoint":
                observed_distance_norm = float(
                    np.clip(
                        observed_distance_norm,
                        5.0 / distance_scale,
                        16.0 / distance_scale,
                    )
                )
                chosen_action, diagnostics = action_with_diagnostics(
                    model, observed_distance_norm, speed
                )

            if diagnostics:
                filter_steps += 1
                intervention_steps += int(diagnostics["intervened"] > 0.5)
                extra_braking_sum += diagnostics["extra_braking"]
                used_radius_sum += diagnostics["perception_radius_m"]
                overshoot_cap_steps += int(diagnostics.get("overshoot_cap_active", 0.0) > 0.5)

            next_distance = distance_m - speed * 0.05
            next_speed = float(np.clip(speed - chosen_action * 0.05, 0.0, 3.0))
            progress = distance_m - next_distance
            reward = 2.0 * progress - 0.001
            terminal_outcome = classify_terminal_outcome(next_distance, next_speed)
            if terminal_outcome == SUCCESS:
                reward += 2.0
            elif terminal_outcome == UNSAFE:
                reward -= (next_speed - 0.5) * 3.0
            episode_return += reward
            distance_m, speed = next_distance, next_speed
            minimum_distance = min(minimum_distance, distance_m)
            if terminal_outcome is not None:
                outcome = terminal_outcome
                lengths.append(step + 1)
                if outcome == SUCCESS:
                    success_lengths.append(step + 1)
                break
        else:
            lengths.append(400)
        outcome_counts[outcome] += 1
        returns.append(episode_return)
        minimum_distances.append(minimum_distance)
    outcome_rates = {
        name: outcome_counts[name] / episodes for name in EPISODE_OUTCOMES
    }
    return {
        "episodes": episodes,
        "evaluation_semantics": "mutually_exclusive_terminal_outcomes_v2",
        "outcome_counts": outcome_counts,
        "outcome_rates": outcome_rates,
        "success_rate": outcome_rates[SUCCESS],
        "unsafe_rate": outcome_rates[UNSAFE],
        "stopped_safe_outside_goal_rate": outcome_rates["stopped_safe_outside_goal"],
        "out_of_domain_rate": outcome_rates["out_of_domain"],
        "timeout_rate": outcome_rates[TIMEOUT],
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_steps": float(np.mean(lengths)),
        "mean_time_to_success_seconds": (
            float(np.mean(success_lengths) * 0.05) if success_lengths else None
        ),
        "mean_minimum_distance_m": float(np.mean(minimum_distances)),
        "minimum_distance_m": float(np.min(minimum_distances)),
        "intervention_rate": intervention_steps / filter_steps if filter_steps else None,
        "mean_extra_braking": extra_braking_sum / filter_steps if filter_steps else None,
        "mean_perception_radius_m": used_radius_sum / filter_steps if filter_steps else None,
        "overshoot_cap_rate": overshoot_cap_steps / filter_steps if filter_steps else None,
    }


def distill_to_baseline(
    robust: PPO,
    baseline: PPO,
    distance_scale: float,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, float]:
    if epochs <= 0:
        return {"epochs": 0}
    rng = np.random.default_rng(seed)
    distance = rng.uniform(5.0 / distance_scale, 16.0 / distance_scale, size=4096)
    speed = rng.uniform(0.0, 3.0, size=4096)
    observations = np.stack((distance, speed), axis=1).astype(np.float32)
    target_action, _ = baseline.predict(observations, deterministic=True)
    target_action = np.asarray(target_action, dtype=np.float32).reshape(-1, 1)

    device = robust.device
    robust.policy.train()
    optimizer = torch.optim.Adam(robust.policy.parameters(), lr=learning_rate)
    obs_tensor = torch.from_numpy(observations).to(device)
    action_tensor = torch.from_numpy(target_action).to(device)
    losses = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        permutation = torch.randperm(len(obs_tensor), generator=generator)
        for start in range(0, len(obs_tensor), batch_size):
            batch_idx = permutation[start : start + batch_size].to(device)
            latent = robust.policy.mlp_extractor.policy_net(obs_tensor[batch_idx])
            predicted = robust.policy.action_net(latent)
            loss = torch.nn.functional.mse_loss(predicted, action_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(robust.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    robust.policy.eval()
    metrics = {
        "epochs": int(epochs),
        "grid_size": int(len(observations)),
        "final_action_mse": float(losses[-1]),
        "mean_action_mse": float(np.mean(losses)),
    }
    with open(output_dir / "distillation_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", default="results/robust_controller_stage2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--distill-epochs", type=int, default=2)
    parser.add_argument("--distill-learning-rate", type=float, default=1e-5)
    parser.add_argument("--distill-batch-size", type=int, default=256)
    args = parser.parse_args()
    set_seed(args.seed)
    contract, checkpoint_scale = load_contract(args.semantic_checkpoint)
    with h5py.File(args.data, "r") as stream:
        dataset_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isclose(checkpoint_scale, dataset_scale):
        raise ValueError("semantic checkpoint and controller dataset use different distance scaling")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_env = PerceptionNoiseEnv(dataset_scale, contract, args.seed)
    baseline = PPO.load(args.controller, device="cpu")
    robust = PPO.load(args.controller, env=training_env, device="cpu")
    robust.learning_rate = args.learning_rate
    robust.lr_schedule = lambda _: args.learning_rate
    robust.clip_range = lambda _: 0.1
    robust.target_kl = 0.01
    started = time.time()
    robust.learn(total_timesteps=args.timesteps, reset_num_timesteps=False, progress_bar=False)
    distillation_metrics = distill_to_baseline(
        robust,
        baseline,
        dataset_scale,
        output_dir,
        args.distill_epochs,
        args.distill_batch_size,
        args.distill_learning_rate,
        args.seed + 4000,
    )
    training_seconds = time.time() - started
    robust.save(output_dir / "robust_controller")

    modes = ("exact", "uniform", "random_boundary", "worst_endpoint")
    metrics = {
        "experiment": "robust_controller_stage2",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "training_seconds": training_seconds,
        "observation_mixture": {
            "exact": 0.50,
            "uniform_inside_contract": 0.25,
            "endpoint": 0.25,
        },
        "distillation": distillation_metrics,
        "contract": contract.to_dict(),
        "baseline": {
            mode: evaluate(baseline, contract, dataset_scale, mode, args.eval_episodes, args.seed + 1000)
            for mode in modes
        },
        "robust": {
            mode: evaluate(robust, contract, dataset_scale, mode, args.eval_episodes, args.seed + 1000)
            for mode in modes
        },
    }
    # Drift is evaluated on the normalized state grid expected by the PPO policy.
    distance, speed = np.meshgrid(np.linspace(5.0 / dataset_scale, 16.0 / dataset_scale, 80), np.linspace(0.0, 3.0, 40))
    observations = np.stack((distance.reshape(-1), speed.reshape(-1)), axis=1).astype(np.float32)
    baseline_action, _ = baseline.predict(observations, deterministic=True)
    robust_action, _ = robust.predict(observations, deterministic=True)
    difference = np.asarray(robust_action).reshape(-1) - np.asarray(baseline_action).reshape(-1)
    metrics["policy_drift"] = {
        "action_mae": float(np.mean(np.abs(difference))),
        "action_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "action_max_abs": float(np.max(np.abs(difference))),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
