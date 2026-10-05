"""Prepare aligned representation cache, train fresh PPO, run matched diagnostic.

Invoked by the user. Test images are neither encoded nor evaluated.
Re-run the command to resume completed stages after checking their input hashes.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.dino_latent.encoder import DinoSafetyLatentEncoder
from Aebs.dino_latent.holdout_experiment import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-checkpoint", type=Path,
                        default=Path("results/dino_improvement/05_representation_alignment_ab/B_joint/dino_safety_latent.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("results/dino_improvement/split.json"))
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_improvement/06_B_joint_ppo"))
    parser.add_argument("--timesteps", type=int, default=200000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.timesteps < 1:
        parser.error("timesteps must be positive")
    old = json.loads(args.manifest.read_text())
    rep = torch.load(args.representation_checkpoint, map_location="cpu")
    if rep.get("manifest_sha256") != sha256(args.manifest):
        parser.error("representation was not trained on this manifest")
    if old["hashes"]["data"] != sha256(args.data):
        parser.error("dataset changed")
    if rep["dino_weights_sha256"] != sha256(Path(rep["dino_weights"])):
        parser.error("DINO weights changed")
    registration = {"representation": str(args.representation_checkpoint),
                    "representation_sha256": sha256(args.representation_checkpoint),
                    "source_manifest_sha256": sha256(args.manifest), "data_sha256": sha256(args.data),
                    "timesteps": args.timesteps, "seed": 7, "test_used": False,
                    "distance_scale_m": float(rep["distance_scale_m"])}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / "config.json"
    if config.exists():
        if json.loads(config.read_text()) != registration:
            parser.error("existing run has different inputs; choose a new output-dir")
    else:
        if any(args.output_dir.iterdir()):
            parser.error("unregistered nonempty output directory")
        config.write_text(json.dumps(registration, indent=2))
    cache = args.output_dir / "latent_data.npz"
    manifest_path = args.output_dir / "split.json"
    if not manifest_path.exists():
        with h5py.File(args.data, "r") as data:
            n = len(data["y_train"])
            ids = np.asarray(sorted(old["indices"]["train"] + old["indices"]["validation"]), dtype=int)
            if sorted(sum(old["indices"].values(), [])) != list(range(n)):
                parser.error("invalid split indices")
            images = np.asarray(data["X_train"][ids], dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = DinoSafetyLatentEncoder(args.representation_checkpoint, device).to(device).eval()
        latent = np.zeros((n, 32), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(ids), 32):
                batch = torch.from_numpy(images[start:start+32]).to(device)
                latent[ids[start:start+32]] = encoder(batch).cpu().numpy()
        if not np.isfinite(latent).all():
            raise ValueError("nonfinite cached latent")
        np.savez_compressed(cache, latent=latent, available_indices=ids)
        new = dict(old)
        new.update({"hashes": {"data": sha256(args.data), "representation": sha256(args.representation_checkpoint),
                               "latent_data": sha256(cache)},
                    "scope": "representation retrained on train, selected on historical validation; test not encoded",
                    "source_manifest_sha256": sha256(args.manifest), "test_encoded": False})
        manifest_path.write_text(json.dumps(new, indent=2))
        print("[Aligned latent cache] encoded=%d; test=0; scale=%.8f" % (len(ids), rep["distance_scale_m"]), flush=True)
    registered = json.loads(manifest_path.read_text())
    if registered["hashes"] != {"data": sha256(args.data), "representation": sha256(args.representation_checkpoint), "latent_data": sha256(cache)}:
        parser.error("cached inputs changed")
    shared = ["--representation-checkpoint", str(args.representation_checkpoint), "--data", str(args.data),
              "--manifest", str(manifest_path), "--latent-data", str(cache)]
    ppo_dir = args.output_dir / "ppo"
    diag_dir = args.output_dir / "matched"
    if not (ppo_dir / "metrics.json").exists():
        subprocess.run([sys.executable, "-m", "Aebs.dino_latent.holdout_experiment", "train"] + shared +
                       ["--skip-heldout-rollout", "--timesteps", str(args.timesteps), "--output-dir", str(ppo_dir)], check=True)
    if not (diag_dir / "metrics.json").exists():
        subprocess.run([sys.executable, "-m", "Aebs.dino_latent.diagnose_matched_images"] + shared +
                       ["--controller", str(ppo_dir / "latent_ppo.zip"), "--output-dir", str(diag_dir)], check=True)
    ppo = json.loads((ppo_dir / "metrics.json").read_text())
    diag = json.loads((diag_dir / "metrics.json").read_text())
    controller_hash = sha256(ppo_dir / "latent_ppo.zip")
    for result in (ppo, diag):
        if result["hashes"] != registered["hashes"] or result["manifest_sha256"] != sha256(manifest_path):
            parser.error("saved results do not match this representation/cache")
    if ppo["checkpoint_sha256"] != controller_hash or diag["controller_sha256"] != controller_hash:
        parser.error("saved results do not match saved PPO")
    print("[B representation -> fresh PPO -> matched diagnostic]")
    print("test not encoded/evaluated; sparse validation rollout skipped")
    for name, value in ppo["evaluation"].items():
        print(name, {k: value[k] for k in ("success_rate", "unsafe_rate", "stopped_safe_outside_goal_rate", "timeout_rate", "mean_steps")})
    for name, value in diag["subsets"].items():
        print(name, {k: value[k] for k in ("checked_nonterminal_pairs", "action_mae", "action_max_abs", "image_only_next_unsafe", "surrogate_only_next_unsafe", "both_next_unsafe")})
    print("PPO metrics:", ppo_dir / "metrics.json")
    print("matched metrics:", diag_dir / "metrics.json")
    print("No end-to-end safety claim; SBC has not been retrained.")


if __name__ == "__main__":
    main()
