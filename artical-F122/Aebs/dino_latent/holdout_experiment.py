"""Grouped image-library holdout for PPO; old representation remains fixed.

This is a development diagnostic, not an end-to-end unseen-data claim.
No CP, SBC training, or QP is invoked here.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from Aebs.dino_latent.train_latent_ppo import LatentAebsEnv, evaluate, print_result, set_seed


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_split(distances, seed=7, width=0.25):
    if width <= 0:
        raise ValueError("group width must be positive")
    groups = np.floor((distances.astype(np.float64) - float(distances.min())) / width + 1e-6).astype(int)
    unique = np.unique(groups)
    if len(unique) < 5:
        raise ValueError("need at least five distance groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    n_holdout = max(1, int(round(len(unique) * 0.2)))
    partitions = {
        "test": shuffled[:n_holdout],
        "validation": shuffled[n_holdout:2 * n_holdout],
        "train": shuffled[2 * n_holdout:],
    }
    return {
        name: np.flatnonzero(np.isin(groups, selected)).tolist()
        for name, selected in partitions.items()
    }, groups


class Progress(BaseCallback):
    def _on_step(self):
        if self.num_timesteps % 20000 == 0:
            print("PPO progress: %d steps" % self.num_timesteps, flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "evaluate", "train"])
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--representation-checkpoint", type=Path,
                        default=Path("results/dino_safety_latent_stage1/dino_safety_latent.pt"))
    parser.add_argument("--latent-data", type=Path,
                        default=Path("results/dino_safety_latent_stage1/latent_data.npz"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/dino_improvement/split.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--controller", type=Path,
                        default=Path("results/dino_latent_ppo_stage2_mixed_20260928/latent_ppo.zip"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-width", type=float, default=0.25)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-grid-size", type=int, default=20)
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")
    parser.add_argument("--skip-heldout-rollout", action="store_true",
                        help="Evaluate surrogate/train rollouts only; use matched-image diagnostic for validation")
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_seed(args.seed)
    if args.timesteps <= 0 or args.eval_grid_size < 2:
        parser.error("timesteps must be positive and grid size at least 2")
    with h5py.File(args.data, "r") as data:
        distances = np.asarray(data["y_train"], dtype=np.float32).reshape(-1)
    hashes = {"data": sha256(args.data), "representation": sha256(args.representation_checkpoint),
              "latent_data": sha256(args.latent_data)}
    with np.load(args.latent_data) as data:
        if data["latent"].shape != (len(distances), 32) or not np.isfinite(data["latent"]).all():
            raise ValueError("latent cache must contain one finite 32-D row per image")
        available = set(data["available_indices"].tolist()) if "available_indices" in data else None

    if args.mode == "prepare":
        if args.manifest.exists():
            parser.error("manifest already exists; preserve it or choose a new path")
        split, groups = grouped_split(distances, args.seed, args.group_width)
        manifest = {
            "seed": args.seed, "group_width_m": args.group_width,
            "groups": groups.tolist(), "indices": split, "hashes": hashes,
            "scope": "PPO image-library holdout only; representation was previously trained",
            "limitation": "same historical dataset; not new scenes or end-to-end held-out data",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("[DINO grouped split prepared]")
        for name, indices in split.items():
            print("%s: images=%d groups=%d" % (name, len(indices), len(set(groups[indices]))))
        print("image_overlap=0, distance_group_overlap=0")
        print("manifest: %s" % args.manifest)
        return

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["hashes"] != hashes:
        parser.error("dataset/representation/cache changed since split registration")
    indices = manifest["indices"]
    needed = indices["train"] + ([] if args.skip_heldout_rollout else indices[args.eval_split])
    if available is not None and not set(needed).issubset(available):
        parser.error("requested split is unavailable in this latent cache")
    flat = [i for part in indices.values() for i in part]
    if sorted(flat) != list(range(len(distances))):
        parser.error("split must partition all image indices without overlap")
    groups = np.asarray(manifest["groups"])
    group_sets = [set(groups[indices[name]]) for name in ("train", "validation", "test")]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        parser.error("distance groups overlap")
    if args.output_dir is None:
        parser.error("--output-dir is required for evaluate/train")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose another")
    if args.mode == "train" and args.eval_split != "validation":
        parser.error("train evaluates validation only; use evaluate for the locked test")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "mode": args.mode, "seed": args.seed, "requested_timesteps": args.timesteps,
        "hashes": hashes, "manifest_sha256": sha256(args.manifest),
        "eval_split": args.eval_split, "eval_grid_size": args.eval_grid_size,
        "heldout_rollout_skipped": args.skip_heldout_rollout,
        "representation_scope": manifest["scope"],
        "image_counts": {name: len(value) for name, value in indices.items()},
        "note": "An old controller may already have seen these images. Test subset evaluation does not erase prior exposure.",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    def env(mode, subset=None):
        return LatentAebsEnv(
            args.representation_checkpoint, mode, data_path=args.data,
            latent_data_path=args.latent_data,
            image_indices=indices[subset] if subset else None,
        )

    started = time.time()
    if args.mode == "train":
        model = PPO(
            "MlpPolicy", env("mixed_episode", "train"), verbose=0,
            learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
            seed=args.seed, device="cpu",
        )
        print("Training fresh PPO: image lookup restricted to train groups", flush=True)
        model.learn(total_timesteps=args.timesteps, callback=Progress())
        model.save(args.output_dir / "latent_ppo.zip")
        checkpoint = args.output_dir / "latent_ppo.zip"
    else:
        checkpoint = args.controller
        model = PPO.load(checkpoint, device="cpu")
    result = dict(config)
    result["checkpoint"] = str(checkpoint)
    result["checkpoint_sha256"] = sha256(checkpoint)
    result["actual_model_timesteps"] = int(model.num_timesteps)
    result["evaluation"] = {}
    paths = [
        ("surrogate", "surrogate", None),
        ("image_train", "image_nearest", "train"),
    ]
    if not args.skip_heldout_rollout:
        paths.append(("image_" + args.eval_split, "image_nearest", args.eval_split))
    for name, mode, subset in paths:
        print("Evaluating %s ..." % name, flush=True)
        metrics = evaluate(model, env(mode, subset), args.eval_grid_size)
        result["evaluation"][name] = metrics
        print_result(name, metrics)
    s = result["evaluation"]["surrogate"]
    v = result["evaluation"].get("image_" + args.eval_split)
    result["development_gate_pass"] = (
        s["success_rate"] == 1.0 and s["unsafe_rate"] == 0.0
        and v["success_rate"] >= 0.95 and v["unsafe_rate"] == 0.0
    ) if v is not None else None
    result["runtime_seconds"] = time.time() - started
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print("[DINO PPO image-library holdout: development diagnostic]")
    print("representation_is_newly_heldout=False")
    print("development_gate_pass=%s" % result["development_gate_pass"])
    for name, metrics in result["evaluation"].items():
        print_result(name, metrics)
        print("%s lookup_distance_error_m=%s" % (name, metrics["nearest_image_distance_error_m"]))
    print("checkpoint: %s" % checkpoint)
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
