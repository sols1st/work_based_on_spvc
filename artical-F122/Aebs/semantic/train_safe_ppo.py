"""Train and select a standalone safety-oriented PPO controller.

This experiment does not load or call a safety filter. Perception-error
randomization is used during training, and checkpoints are selected by
closed-loop safety outcomes rather than mean episode reward.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from Aebs.semantic.robust_controller import evaluate, load_contract
from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import (
    OUT_OF_DOMAIN,
    STOPPED_SAFE_OUTSIDE_GOAL,
    SUCCESS,
    TIMEOUT,
    UNSAFE,
)


MODES = ("exact", "uniform", "random_boundary", "worst_endpoint")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SafePPOEnv(AebsEnv):
    """AEBS training environment with noisy observations and dense safety reward.

    The simulator state always remains the true state. Only the observation
    passed to PPO is perturbed. One error regime is held for an episode so the
    policy sees persistent bias as well as independent uniform noise.
    """

    NOISE_MODES = ("exact", "uniform", "random_boundary", "lower", "upper")
    NOISE_PROBABILITIES = np.array((0.20, 0.30, 0.20, 0.15, 0.15))

    def __init__(
        self,
        distance_scale: float,
        contract,
        seed: int = 0,
        max_episode_steps: int = 400,
        unsafe_penalty: float = 200.0,
        success_bonus: float = 100.0,
        failure_penalty: float = 50.0,
        risk_weight: float = 25.0,
        tracking_weight: float = 0.5,
        nominal_deceleration: float = 1.5,
    ):
        super().__init__(distance_scale, max_episode_steps=max_episode_steps)
        self.contract = contract
        self.rng = np.random.default_rng(seed)
        self.unsafe_penalty = float(unsafe_penalty)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.risk_weight = float(risk_weight)
        self.tracking_weight = float(tracking_weight)
        self.nominal_deceleration = float(nominal_deceleration)
        self.noise_mode = "exact"
        self.previous_action = 0.0

    def _radius_norm(self, distance_m: float) -> float:
        return float(self.contract.radius(np.array([distance_m]))[0])

    def _observe(self, true_observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(true_observation, dtype=np.float32).copy()
        radius = self._radius_norm(float(observation[0] * self.std1))
        if self.noise_mode == "uniform":
            error = self.rng.uniform(-radius, radius)
        elif self.noise_mode == "random_boundary":
            error = radius * self.rng.choice((-1.0, 1.0))
        elif self.noise_mode == "lower":
            error = -radius
        elif self.noise_mode == "upper":
            error = radius
        else:
            error = 0.0
        observation[0] = np.clip(
            observation[0] + error,
            self.observation_space.low[0],
            self.observation_space.high[0],
        )
        return observation

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.noise_mode = str(
            self.rng.choice(self.NOISE_MODES, p=self.NOISE_PROBABILITIES)
        )
        distance_m = float(self.rng.uniform(15.0, 16.0))
        speed_mps = float(self.rng.uniform(2.5, 3.0))
        self.state = np.array(
            [distance_m / self.std1, speed_mps], dtype=np.float32
        )
        self.elapsed_steps = 0
        self.previous_action = 0.0
        return self._observe(self.state), {
            "noise_mode": self.noise_mode,
            "true_observation": self.state.copy(),
        }

    def _dense_safety_reward(self, distance_m: float, speed_mps: float) -> float:
        # The recoverability envelope uses the physical maximum braking (3 m/s²).
        remaining = max(distance_m - 6.0, 0.0)
        safe_speed = np.sqrt(0.5**2 + 2.0 * 3.0 * remaining)
        overspeed = max(speed_mps - safe_speed, 0.0)

        # Track a gentler profile aimed at 0.45 m/s at the 6 m goal boundary.
        # It remains positive up to the boundary, avoiding an early full stop.
        target_remaining = max(distance_m - 6.0, 0.0)
        target_speed = min(
            3.0,
            np.sqrt(0.45**2 + 2.0 * self.nominal_deceleration * target_remaining),
        )
        tracking_error = (speed_mps - target_speed) / 3.0
        return -self.risk_weight * overspeed**2 - self.tracking_weight * tracking_error**2

    def step(self, action):
        clipped_action = float(np.clip(np.asarray(action).reshape(-1)[0], -3.0, 3.0))
        _, base_reward, terminated, truncated, info = super().step(
            np.array([clipped_action], dtype=np.float32)
        )
        distance_m = float(self.state[0] * self.std1)
        speed_mps = float(self.state[1])
        reward = float(base_reward) + self._dense_safety_reward(distance_m, speed_mps)
        reward -= 0.002 * clipped_action**2
        reward -= 0.001 * (clipped_action - self.previous_action) ** 2
        self.previous_action = clipped_action

        outcome = info.get("outcome")
        if outcome == SUCCESS:
            # Replace the small legacy terminal bonus with a decisive success bonus.
            reward += self.success_bonus - 2.0
        elif outcome == UNSAFE:
            reward -= self.unsafe_penalty
        elif outcome in (STOPPED_SAFE_OUTSIDE_GOAL, OUT_OF_DOMAIN, TIMEOUT):
            reward -= self.failure_penalty

        info.update(
            {
                "noise_mode": self.noise_mode,
                "true_observation": self.state.copy(),
                "perception_radius_norm": self._radius_norm(distance_m),
            }
        )
        return self._observe(self.state), reward, terminated, truncated, info


def safety_rank(evaluations: Dict[str, Dict[str, object]]) -> Tuple[float, ...]:
    """Lexicographic rank: unsafe first, then all other failures and efficiency."""

    unsafe = sum(float(evaluations[mode]["unsafe_rate"]) for mode in MODES)
    non_success = sum(1.0 - float(evaluations[mode]["success_rate"]) for mode in MODES)
    timeout = sum(float(evaluations[mode]["timeout_rate"]) for mode in MODES)
    mean_steps = float(np.mean([evaluations[mode]["mean_steps"] for mode in MODES]))
    negative_return = -float(
        np.mean([evaluations[mode]["mean_return"] for mode in MODES])
    )
    return unsafe, non_success, timeout, mean_steps, negative_return


def evaluate_modes(
    model: PPO,
    contract,
    distance_scale: float,
    episodes: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    return {
        mode: evaluate(model, contract, distance_scale, mode, episodes, seed)
        for mode in MODES
    }


class SafetyEvaluationCallback(BaseCallback):
    """Save the checkpoint with the best closed-loop safety rank."""

    def __init__(
        self,
        contract,
        distance_scale: float,
        output_dir: Path,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
    ):
        super().__init__(verbose=1)
        self.contract = contract
        self.distance_scale = float(distance_scale)
        self.output_dir = output_dir
        self.eval_freq = max(int(eval_freq), 1)
        self.eval_episodes = int(eval_episodes)
        self.seed = int(seed)
        self.best_rank = None
        self.history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        evaluations = evaluate_modes(
            self.model,
            self.contract,
            self.distance_scale,
            self.eval_episodes,
            # Keep the episode set fixed so checkpoint ranks are comparable.
            self.seed,
        )
        rank = safety_rank(evaluations)
        record = {
            "timesteps": int(self.num_timesteps),
            "rank": list(rank),
            "evaluation": evaluations,
        }
        self.history.append(record)
        if self.best_rank is None or rank < self.best_rank:
            self.best_rank = rank
            self.model.save(self.output_dir / "safe_ppo")
            record["saved_as_best"] = True
        with open(self.output_dir / "training_evaluations.json", "w", encoding="utf-8") as stream:
            json.dump(self.history, stream, indent=2, ensure_ascii=False)
        print(
            f"safety-eval steps={self.num_timesteps} "
            f"unsafe_sum={rank[0]:.4f} non_success_sum={rank[1]:.4f} "
            f"timeout_sum={rank[2]:.4f} best={self.best_rank}",
            flush=True,
        )
        return True


def train(args: argparse.Namespace) -> Dict[str, object]:
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract, distance_scale = load_contract(args.semantic_checkpoint)

    def make_environment(rank: int):
        return SafePPOEnv(
            distance_scale,
            contract,
            seed=args.seed + rank,
            unsafe_penalty=args.unsafe_penalty,
            success_bonus=args.success_bonus,
            failure_penalty=args.failure_penalty,
            risk_weight=args.risk_weight,
            tracking_weight=args.tracking_weight,
            nominal_deceleration=args.nominal_deceleration,
        )

    training_env = DummyVecEnv(
        [lambda rank=rank: make_environment(rank) for rank in range(args.n_envs)]
    )
    training_env.seed(args.seed)
    policy_kwargs = {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}
    if args.init_model:
        model = PPO.load(args.init_model, env=training_env, device=args.device)
        model.learning_rate = args.learning_rate
        model.lr_schedule = lambda _: args.learning_rate
    else:
        model = PPO(
            "MlpPolicy",
            training_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.003,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )

    # Callback calls occur once per vector step, so convert environment timesteps.
    callback_frequency = max(args.eval_freq // args.n_envs, 1)
    checkpoint_frequency = max(args.checkpoint_freq // args.n_envs, 1)
    safety_callback = SafetyEvaluationCallback(
        contract,
        distance_scale,
        output_dir,
        callback_frequency,
        args.selection_episodes,
        args.seed + 10_000,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="safe_ppo",
    )
    started = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, safety_callback],
        reset_num_timesteps=not bool(args.init_model),
        progress_bar=False,
    )
    training_seconds = time.time() - started
    final_path = output_dir / "final_ppo"
    model.save(final_path)
    training_env.close()

    best_path = output_dir / "safe_ppo.zip"
    if not best_path.exists():
        # Covers deliberately tiny smoke runs whose eval interval was not reached.
        model.save(output_dir / "safe_ppo")
    best_model = PPO.load(best_path, device=args.device)
    final_evaluation = evaluate_modes(
        best_model,
        contract,
        distance_scale,
        args.eval_episodes,
        args.seed + 20_000,
    )
    metrics = {
        "experiment": "standalone_safe_ppo_online_rl",
        "deployment_controller": "ppo_without_safety_filter",
        "safety_filter_used": False,
        "seed": int(args.seed),
        "timesteps": int(args.timesteps),
        "n_envs": int(args.n_envs),
        "training_seconds": float(training_seconds),
        "checkpoint": str(best_path),
        "final_training_checkpoint": str(final_path) + ".zip",
        "selection_rule": "lexicographic(unsafe, non_success, timeout, mean_steps, -mean_return)",
        "best_selection_rank": (
            list(safety_callback.best_rank)
            if safety_callback.best_rank is not None
            else None
        ),
        "final_evaluation_rank": list(safety_rank(final_evaluation)),
        "reward": {
            "unsafe_penalty": args.unsafe_penalty,
            "success_bonus": args.success_bonus,
            "failure_penalty": args.failure_penalty,
            "risk_weight": args.risk_weight,
            "tracking_weight": args.tracking_weight,
            "nominal_deceleration": args.nominal_deceleration,
        },
        "training_observation_mixture": dict(
            zip(SafePPOEnv.NOISE_MODES, SafePPOEnv.NOISE_PROBABILITIES.tolist())
        ),
        "evaluation": final_evaluation,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument("--output-dir", default="results/mvp/02_safe_ppo_online")
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--selection-episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--unsafe-penalty", type=float, default=200.0)
    parser.add_argument("--success-bonus", type=float, default=100.0)
    parser.add_argument("--failure-penalty", type=float, default=50.0)
    parser.add_argument("--risk-weight", type=float, default=25.0)
    parser.add_argument("--tracking-weight", type=float, default=0.5)
    parser.add_argument("--nominal-deceleration", type=float, default=1.5)
    parser.add_argument("--device", default="auto")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
