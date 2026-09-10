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
        # Boundary sampling trains against the full contract instead of only likely errors.
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
    prediction, _ = model.predict(np.array([distance_norm, speed], dtype=np.float32), deterministic=True)
    return float(np.clip(np.asarray(prediction).reshape(-1)[0], -3.0, 3.0))


def evaluate(
    model: PPO,
    contract: StateConditionalErrorContract,
    distance_scale: float,
    mode: str,
    episodes: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    successes, unsafe_episodes, returns, lengths = 0, 0, [], []
    for _ in range(episodes):
        distance_m = float(rng.uniform(15.0, 16.0))
        speed = float(rng.uniform(2.5, 3.0))
        episode_return = 0.0
        unsafe = False
        success = False
        for step in range(400):
            distance_norm = distance_m / distance_scale
            radius = float(contract.radius(np.array([distance_m]))[0])
            if mode == "exact":
                chosen_action = action(model, distance_norm, speed)
            elif mode == "uniform":
                chosen_action = action(model, distance_norm + rng.uniform(-radius, radius), speed)
            elif mode == "random_boundary":
                chosen_action = action(model, distance_norm + radius * rng.choice((-1.0, 1.0)), speed)
            elif mode == "worst_endpoint":
                # Positive action is braking in these dynamics; the smaller endpoint action is safety-worst.
                chosen_action = min(
                    action(model, distance_norm - radius, speed),
                    action(model, distance_norm + radius, speed),
                )
            else:
                raise ValueError(f"unknown evaluation mode: {mode}")

            next_distance = distance_m - speed * 0.05
            next_speed = float(np.clip(speed - chosen_action * 0.05, 0.0, 3.0))
            progress = distance_m - next_distance
            reward = 2.0 * progress - 0.001
            if next_distance <= 6.0:
                if next_speed <= 0.5:
                    reward += 2.0
                    success = True
                else:
                    reward -= (next_speed - 0.5) * 3.0
            if 5.0 <= next_distance <= 6.0 and next_speed >= 0.5:
                unsafe = True
            episode_return += reward
            distance_m, speed = next_distance, next_speed
            if next_distance >= 16.0 or next_distance <= 5.0 or next_speed <= 0.0:
                lengths.append(step + 1)
                break
        else:
            lengths.append(400)
        successes += int(success and not unsafe)
        unsafe_episodes += int(unsafe)
        returns.append(episode_return)
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "unsafe_rate": unsafe_episodes / episodes,
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_steps": float(np.mean(lengths)),
    }


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
    training_seconds = time.time() - started
    robust.save(output_dir / "robust_controller")

    modes = ("exact", "uniform", "random_boundary", "worst_endpoint")
    metrics = {
        "experiment": "robust_controller_stage2",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "training_seconds": training_seconds,
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
