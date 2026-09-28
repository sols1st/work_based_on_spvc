"""Train the minimal 33-input PPO for the frozen-DINO safety latent.

No conformal prediction, SBC, or QP is used in this stage.  The AEBS
dynamics, reward, action bounds, and terminal conditions remain those of the
original controller experiment; only the observation is replaced by
[q_psi(distance), speed].
"""

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import h5py
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import TIMEOUT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LatentAebsEnv(gym.Env):
    """Expose the original AEBS environment through a 32-D latent observation."""

    metadata = {"render_modes": []}

    def __init__(self, checkpoint_path, observation_mode="surrogate",
                 data_path=Path("Aebs/data/Downsampled.h5"),
                 latent_data_path=Path("results/dino_safety_latent_stage1/latent_data.npz"),
                 max_episode_steps=400):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.distance_scale_m = float(checkpoint["distance_scale_m"])
        self.q_model = PhysicalToLatent(
            physical_dim=1, latent_dim=int(checkpoint["latent_dimension"])
        )
        self.q_model.load_state_dict(checkpoint["physical_to_latent_state_dict"])
        self.q_model.eval()
        for parameter in self.q_model.parameters():
            parameter.requires_grad_(False)
        self.latent_dimension = int(checkpoint["latent_dimension"])
        self.base = AebsEnv(self.distance_scale_m, max_episode_steps=max_episode_steps)
        self.action_space = self.base.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.latent_dimension + 1,), dtype=np.float32,
        )
        if observation_mode not in {"surrogate", "image_nearest"}:
            raise ValueError("observation_mode must be surrogate or image_nearest")
        self.observation_mode = observation_mode
        self.image_distances_m = None
        self.image_latents = None
        if observation_mode == "image_nearest":
            with h5py.File(data_path, "r") as data_file:
                self.image_distances_m = np.asarray(
                    data_file["y_train"], dtype=np.float32
                ).reshape(-1)
            latent_file = np.load(latent_data_path)
            self.image_latents = np.asarray(latent_file["latent"], dtype=np.float32)

    def _latent(self, distance_norm):
        if self.observation_mode == "surrogate":
            value = torch.tensor([[distance_norm]], dtype=torch.float32)
            with torch.no_grad():
                return self.q_model(value).numpy()[0].astype(np.float32)
        distance_m = float(distance_norm) * self.distance_scale_m
        index = int(np.argmin(np.abs(self.image_distances_m - distance_m)))
        return self.image_latents[index].astype(np.float32)

    def _observation(self, physical_observation):
        distance_norm, speed = physical_observation
        return np.concatenate([
            self._latent(float(distance_norm)), np.asarray([speed], dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        physical_observation, info = self.base.reset(seed=seed, options=options)
        return self._observation(physical_observation), info

    def reset_to(self, distance_m, speed_mps):
        self.base.state = np.asarray(
            [distance_m / self.distance_scale_m, speed_mps], dtype=np.float32
        )
        self.base.elapsed_steps = 0
        return self._observation(self.base.state)

    def step(self, action):
        physical_observation, reward, terminated, truncated, info = self.base.step(action)
        return self._observation(physical_observation), reward, terminated, truncated, info


def evaluate(model, env, grid_size):
    distances = np.linspace(15.0, 16.0, grid_size, dtype=np.float32)
    speeds = np.linspace(2.5, 3.0, grid_size, dtype=np.float32)
    outcomes = Counter()
    returns = []
    steps = []
    for distance_m in distances:
        for speed_mps in speeds:
            observation = env.reset_to(float(distance_m), float(speed_mps))
            episode_return = 0.0
            outcome = None
            for step in range(1, env.base.max_episode_steps + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    outcome = info.get("outcome", TIMEOUT)
                    break
            outcomes[outcome or TIMEOUT] += 1
            returns.append(episode_return)
            steps.append(step)
    episodes = grid_size * grid_size
    result = {
        "episodes": episodes,
        "grid": [grid_size, grid_size],
        "counts": dict(outcomes),
        "success_rate": outcomes["success"] / episodes,
        "unsafe_rate": outcomes["unsafe"] / episodes,
        "timeout_rate": outcomes["timeout"] / episodes,
        "stopped_safe_outside_goal_rate": outcomes["stopped_safe_outside_goal"] / episodes,
        "out_of_domain_rate": outcomes["out_of_domain"] / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_steps": float(np.mean(steps)),
    }
    return result


def print_result(name, result):
    print(
        "%s: success=%.2f%% unsafe=%.2f%% timeout=%.2f%% "
        "stopped_safe=%.2f%% out_of_domain=%.2f%% return=%.3f steps=%.1f" % (
            name,
            100 * result["success_rate"],
            100 * result["unsafe_rate"],
            100 * result["timeout_rate"],
            100 * result["stopped_safe_outside_goal_rate"],
            100 * result["out_of_domain_rate"],
            result["mean_return"], result["mean_steps"],
        ), flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation-checkpoint", type=Path,
        default=Path("results/dino_safety_latent_stage1/dino_safety_latent.pt"),
    )
    parser.add_argument(
        "--latent-data", type=Path,
        default=Path("results/dino_safety_latent_stage1/latent_data.npz"),
    )
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dino_latent_ppo_stage2"))
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-grid-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--train-observation-mode", choices=["surrogate", "image_nearest"],
        default="surrogate",
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    registered = {
        "experiment": "dino_latent_ppo_stage2_no_cp",
        "registered_before_training": True,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "observation": "[32-D latent, speed]",
        "train_observation_mode": args.train_observation_mode,
        "excluded": ["conformal prediction", "SBC", "QP"],
        "unchanged_from_original": [
            "AEBS dynamics", "reward", "action bounds", "terminal conditions",
            "PPO hyperparameters",
        ],
        "evaluation_initial_region": {
            "distance_m": [15.0, 16.0], "speed_mps": [2.5, 3.0],
            "grid": [args.eval_grid_size, args.eval_grid_size],
        },
    }
    with open(args.output_dir / "config.json", "w", encoding="utf-8") as output_file:
        json.dump(registered, output_file, indent=2, ensure_ascii=False)

    train_env = LatentAebsEnv(
        args.representation_checkpoint, args.train_observation_mode,
        data_path=args.data, latent_data_path=args.latent_data,
    )
    model = PPO(
        "MlpPolicy", train_env, verbose=0,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        seed=args.seed, device="cpu",
    )
    started = time.time()
    print("[DINO latent PPO stage 2: no CP/SBC/QP]", flush=True)
    print("training timesteps=%d, observation dimension=33, mode=%s" % (
        args.timesteps, args.train_observation_mode
    ), flush=True)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    training_seconds = time.time() - started
    model.save(args.output_dir / "latent_ppo")

    surrogate_env = LatentAebsEnv(args.representation_checkpoint, "surrogate")
    image_env = LatentAebsEnv(
        args.representation_checkpoint, "image_nearest",
        data_path=args.data, latent_data_path=args.latent_data,
    )
    surrogate_result = evaluate(model, surrogate_env, args.eval_grid_size)
    image_result = evaluate(model, image_env, args.eval_grid_size)
    result = {
        **registered,
        "training_seconds": training_seconds,
        "evaluation": {
            "surrogate_latent": surrogate_result,
            "nearest_real_image_latent": image_result,
        },
        "checkpoint": str(args.output_dir / "latent_ppo.zip"),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    print_result("surrogate_latent", surrogate_result)
    print_result("nearest_real_image_latent", image_result)
    print("training seconds=%.3f" % training_seconds)
    print("checkpoint: %s" % (args.output_dir / "latent_ppo.zip"))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
