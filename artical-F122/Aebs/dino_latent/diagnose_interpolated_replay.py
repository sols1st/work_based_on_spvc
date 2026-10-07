"""Counterfactual replay with distance-interpolated train-image latents.

This keeps the frozen R1 representation and PPO unchanged.  Within each
weather/color train library it linearly interpolates between the two adjacent
label distances instead of selecting one nearest image.  The result measures
how much discrete lookup contributes to the existing replay failures; it is
not a realizable camera observation, a new controller, or a safety claim.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from Aebs.dino_latent.diagnose_expanded_failures import load_fixed_inputs
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.run_expanded_ppo import AppearanceEnv
from Aebs.dino_latent.train_latent_ppo import evaluate


class InterpolatedAppearanceEnv(AppearanceEnv):
    """Interpolate train-image latents along label distance per appearance."""

    def __init__(self, checkpoint, data, latent, rows, appearance):
        super().__init__(checkpoint, data, latent, rows, "image_nearest", appearance)
        self.curves = {}
        for key, pool in self.pools.items():
            order = pool[np.argsort(self.image_distances_m[pool], kind="stable")]
            ordered_distances = self.image_distances_m[order]
            unique_distances, inverse = np.unique(ordered_distances, return_inverse=True)
            averaged_latents = np.zeros(
                (len(unique_distances), self.image_latents.shape[1]), dtype=np.float64
            )
            counts = np.zeros(len(unique_distances), dtype=np.int64)
            for position, group in enumerate(inverse):
                averaged_latents[group] += self.image_latents[order[position]]
                counts[group] += 1
            averaged_latents /= counts[:, None]
            self.curves[key] = (
                unique_distances.astype(np.float64),
                averaged_latents.astype(np.float32),
            )
        self.interpolation_spans_m = []

    def _latent(self, distance_norm):
        distance_m = float(distance_norm) * self.distance_scale_m
        distances, latents = self.curves[self.appearance]
        upper = int(np.searchsorted(distances, distance_m, side="left"))
        if upper == 0:
            lower = upper = 0
        elif upper == len(distances):
            lower = upper = len(distances) - 1
        elif np.isclose(distances[upper], distance_m, rtol=0.0, atol=1e-7):
            lower = upper
        else:
            lower = upper - 1
        lower_distance = float(distances[lower])
        upper_distance = float(distances[upper])
        self.lookup_errors_m.append(
            min(abs(distance_m - lower_distance), abs(upper_distance - distance_m))
        )
        self.interpolation_spans_m.append(upper_distance - lower_distance)
        if lower == upper:
            return latents[lower].copy()
        weight = (distance_m - lower_distance) / (upper_distance - lower_distance)
        return ((1.0 - weight) * latents[lower] + weight * latents[upper]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument(
        "--representation", type=Path,
        default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/dino_safety_latent.pt"),
    )
    parser.add_argument("--ppo-dir", type=Path, default=Path("results/dino_expanded_v1/04_R1_ppo"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_expanded_v1/06_interpolated_replay"))
    parser.add_argument("--eval-grid-size", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists() or args.eval_grid_size < 2:
        parser.error("Choose a new output directory and grid size of at least 2")
    fixed = load_fixed_inputs(args, parser)
    model = PPO.load(fixed["controller_path"], device="cpu")
    probe = AppearanceEnv(
        args.representation, fixed["data_path"], fixed["latent_path"],
        fixed["rows"], "image_nearest",
    )
    appearances = sorted(probe.pools)
    results = {}
    comparison = {}
    for appearance in appearances:
        env = InterpolatedAppearanceEnv(
            args.representation, fixed["data_path"], fixed["latent_path"],
            fixed["rows"], appearance,
        )
        result = evaluate(model, env, args.eval_grid_size)
        result["nearest_endpoint_distance_m"] = result.pop(
            "nearest_image_distance_error_m"
        )
        result["interpolation_span_m"] = {
            "mean": float(np.mean(env.interpolation_spans_m)),
            "max": float(np.max(env.interpolation_spans_m)),
        }
        results[appearance] = result
        source = fixed["source_metrics"]["evaluation"][appearance]
        nearest_unsafe = source.get("counts", {}).get("unsafe", 0)
        interpolated_unsafe = result["counts"].get("unsafe", 0)
        comparison[appearance] = {
            "nearest_unsafe_count": nearest_unsafe,
            "interpolated_unsafe_count": interpolated_unsafe,
            "unsafe_count_change": interpolated_unsafe - nearest_unsafe,
        }
        print(
            "interpolated %s: unsafe=%d (nearest=%d)" %
            (appearance, interpolated_unsafe, nearest_unsafe), flush=True
        )
    output = {
        "scope": "fixed-R1/fixed-PPO counterfactual interpolation diagnostic; no training or test",
        "test_used": False,
        "not_online_image_observation": True,
        "cache_manifest_sha256": sha256(fixed["cache_path"]),
        "representation_sha256": sha256(args.representation),
        "controller_sha256": sha256(fixed["controller_path"]),
        "source_metrics_sha256": sha256(fixed["metrics_path"]),
        "eval_grid_size": args.eval_grid_size,
        "evaluation": results,
        "nearest_vs_interpolated": comparison,
        "total_nearest_unsafe": sum(row["nearest_unsafe_count"] for row in comparison.values()),
        "total_interpolated_unsafe": sum(row["interpolated_unsafe_count"] for row in comparison.values()),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[Expanded interpolated replay — fixed model, no training/test]")
    print(
        "total unsafe: nearest=%d interpolated=%d" %
        (output["total_nearest_unsafe"], output["total_interpolated_unsafe"])
    )
    print("metrics:", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
