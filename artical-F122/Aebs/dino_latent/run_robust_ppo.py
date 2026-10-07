"""Train a minimal empirical-residual-robust PPO on frozen R1.

The representation, data split, AEBS dynamics, PPO hyperparameters and seed
remain fixed.  Training episodes use 50% raw train-image nearest replay, 25%
clean q latent, and 25% q latent plus a train-only empirical image-minus-q
residual.  Residual norms are clipped at the train P95 and scaled uniformly
per episode in [0, 1].  Validation and test never define the perturbations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.diagnose_expanded_failures import (
    load_fixed_inputs, scan_matched_failures,
)
from Aebs.dino_latent.diagnose_interpolated_replay import InterpolatedAppearanceEnv
from Aebs.dino_latent.holdout_experiment import Progress
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.run_expanded_ppo import AppearanceEnv
from Aebs.dino_latent.train_latent_ppo import evaluate


class RobustAppearanceEnv(AppearanceEnv):
    """Add a train-only empirical residual path without changing R1."""

    TRAIN_MODE_PROBABILITIES = {
        "image_nearest": 0.50,
        "surrogate": 0.25,
        "residual_corrected": 0.25,
    }

    def __init__(
        self, checkpoint, data, latent, rows, training=False,
        appearance=None, fixed_mode="residual_corrected", fixed_scale=1.0,
    ):
        super().__init__(checkpoint, data, latent, rows, "mixed_episode", appearance)
        if fixed_mode not in self.TRAIN_MODE_PROBABILITIES:
            raise ValueError("invalid fixed robust observation mode")
        if not 0.0 <= fixed_scale <= 1.0:
            raise ValueError("fixed residual scale must be in [0, 1]")
        self.training_mix = bool(training)
        self.fixed_mode = fixed_mode
        self.fixed_scale = float(fixed_scale)
        with torch.no_grad():
            q_at_labels = self.q_model(torch.from_numpy(
                self.image_distances_m[:, None] / self.distance_scale_m
            )).numpy()
        residuals = self.image_latents - q_at_labels
        norms = np.linalg.norm(residuals, axis=1)
        self.residual_radius_p95 = float(np.quantile(norms, 0.95))
        factors = np.minimum(1.0, self.residual_radius_p95 / np.maximum(norms, 1e-12))
        self.clipped_residuals = (residuals * factors[:, None]).astype(np.float32)
        self.active_observation_mode = fixed_mode
        self.residual_scale = self.fixed_scale

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        if self.fixed_appearance is None:
            keys = sorted(self.pools)
            self.appearance = keys[np.random.randint(len(keys))]
        if self.training_mix:
            draw = np.random.random()
            if draw < 0.50:
                self.active_observation_mode = "image_nearest"
            elif draw < 0.75:
                self.active_observation_mode = "surrogate"
            else:
                self.active_observation_mode = "residual_corrected"
            self.residual_scale = float(np.random.random())
        else:
            self.active_observation_mode = self.fixed_mode
            self.residual_scale = self.fixed_scale
        physical_observation, info = self.base.reset(seed=seed, options=options)
        return self._observation(physical_observation), info

    def _latent(self, distance_norm):
        if self.active_observation_mode != "residual_corrected":
            return super()._latent(distance_norm)
        distance_m = float(distance_norm) * self.distance_scale_m
        pool = self.pools[self.appearance]
        local_index = int(pool[np.argmin(np.abs(self.image_distances_m[pool] - distance_m))])
        self.lookup_errors_m.append(
            float(abs(self.image_distances_m[local_index] - distance_m))
        )
        value = torch.tensor([[distance_norm]], dtype=torch.float32)
        with torch.no_grad():
            q_current = self.q_model(value).numpy()[0]
        return (
            q_current + self.residual_scale * self.clipped_residuals[local_index]
        ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/dino_safety_latent.pt"),
    )
    parser.add_argument("--baseline-ppo-dir", type=Path, default=Path("results/dino_expanded_v1/04_R1_ppo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/07_robust_ppo_v1"))
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-grid-size", type=int, default=20)
    parser.add_argument("--speed-count", type=int, default=31)
    args = parser.parse_args()
    if (
        args.output_dir.exists() or args.timesteps < 1
        or args.eval_grid_size < 2 or args.speed_count < 2
    ):
        parser.error("Choose a new output directory and valid positive sizes")
    # Reuse the fully checked baseline adapters; this also proves test_used=False.
    load_args = argparse.Namespace(
        cache_dir=args.cache_dir,
        representation=args.representation,
        ppo_dir=args.baseline_ppo_dir,
    )
    fixed = load_fixed_inputs(load_args, parser)
    torch.set_num_threads(1)

    def appearance_env(mode, appearance=None):
        return AppearanceEnv(
            args.representation, fixed["data_path"], fixed["latent_path"],
            fixed["rows"], mode, appearance,
        )

    training_env = RobustAppearanceEnv(
        args.representation, fixed["data_path"], fixed["latent_path"],
        fixed["rows"], training=True,
    )
    config = {
        "experiment": "frozen_R1_empirical_residual_robust_PPO_v1",
        "registered_before_training": True,
        "representation_sha256": sha256(args.representation),
        "cache_manifest_sha256": sha256(fixed["cache_path"]),
        "baseline_metrics_sha256": sha256(fixed["metrics_path"]),
        "test_used": False,
        "seed": 7,
        "timesteps": args.timesteps,
        "ppo_hyperparameters_unchanged": True,
        "training_mode_probabilities": RobustAppearanceEnv.TRAIN_MODE_PROBABILITIES,
        "residual_source": "train image latent minus q at its own label distance",
        "residual_norm_clip": "train P95",
        "residual_radius_p95": training_env.residual_radius_p95,
        "residual_episode_scale": "uniform [0,1]",
        "representation_frozen": True,
        "excluded": ["test", "representation retraining", "SBC", "QP", "CP"],
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False), encoding="utf-8"
    )
    model = PPO(
        "MlpPolicy", training_env, verbose=0, learning_rate=3e-4,
        n_steps=2048, batch_size=64, n_epochs=10, gamma=.99,
        gae_lambda=.95, ent_coef=.01, seed=7, device="cpu",
    )
    print(
        "Training robust PPO: 50% image, 25% q, 25% q+train residual; "
        "R1 frozen, no test", flush=True,
    )
    model.learn(total_timesteps=args.timesteps, callback=Progress())
    model.save(args.output_dir / "latent_ppo.zip")
    controller_path = args.output_dir / "latent_ppo.zip"

    results = {
        **config,
        "actual_timesteps": int(model.num_timesteps),
        "checkpoint_sha256": sha256(controller_path),
        "evaluation": {"surrogate": evaluate(model, appearance_env("surrogate"), args.eval_grid_size)},
        "matched": {},
    }
    appearances = sorted(appearance_env("image_nearest").pools)
    for appearance in appearances:
        print("Evaluating robust PPO " + appearance, flush=True)
        nearest = evaluate(
            model, appearance_env("image_nearest", appearance), args.eval_grid_size
        )
        interpolated_env = InterpolatedAppearanceEnv(
            args.representation, fixed["data_path"], fixed["latent_path"],
            fixed["rows"], appearance,
        )
        interpolated = evaluate(model, interpolated_env, args.eval_grid_size)
        residual_env = RobustAppearanceEnv(
            args.representation, fixed["data_path"], fixed["latent_path"],
            fixed["rows"], training=False, appearance=appearance,
            fixed_mode="residual_corrected", fixed_scale=1.0,
        )
        residual = evaluate(model, residual_env, args.eval_grid_size)
        results["evaluation"][appearance] = {
            "nearest": nearest,
            "interpolated": interpolated,
            "residual_corrected": residual,
        }
        (args.output_dir / "rollout_metrics.json").write_text(
            json.dumps(results["evaluation"], indent=2, allow_nan=False), encoding="utf-8"
        )
    for split in ("train", "validation"):
        print("Matched robust PPO " + split, flush=True)
        ids = [i for i, row in enumerate(fixed["rows"]) if row["split"] == split]
        summary, image_only, disagreements = scan_matched_failures(
            model, appearance_env("surrogate"), fixed["latents"],
            fixed["distances"], ids, fixed["rows"], args.speed_count,
        )
        results["matched"][split] = {
            **summary,
            "image_only_failures": image_only,
            "outcome_disagreements": disagreements,
        }
        (args.output_dir / "metrics.json").write_text(
            json.dumps(results, indent=2, allow_nan=False), encoding="utf-8"
        )
    print("[Frozen R1 empirical-residual robust PPO — no test/SBC]")
    print("residual_radius_p95=%.9f" % training_env.residual_radius_p95)
    print("surrogate unsafe=%.6f" % results["evaluation"]["surrogate"]["unsafe_rate"])
    for appearance in appearances:
        row = results["evaluation"][appearance]
        print(
            "%s unsafe nearest=%d interpolated=%d residual=%d" % (
                appearance,
                row["nearest"]["counts"].get("unsafe", 0),
                row["interpolated"]["counts"].get("unsafe", 0),
                row["residual_corrected"]["counts"].get("unsafe", 0),
            )
        )
    for split, row in results["matched"].items():
        print(
            "%s matched image_only_unsafe=%d disagreement=%d action_mae=%.6f" % (
                split, row["image_only_next_unsafe"],
                row["outcome_disagreement"], row["action_mae"],
            )
        )
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
