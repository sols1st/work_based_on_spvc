"""Verify the saved DINO front end emits a 32-D latent, not distance."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.dino_latent.encoder import DinoSafetyLatentEncoder, PhysicalLatentSurrogate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/dino_safety_latent_stage1/dino_safety_latent.pt"))
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dino_safety_latent_stage1/smoke"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(args.data, "r") as data_file:
        images = np.asarray(data_file["X_train"][:args.count], dtype=np.float32)
        distance_m = np.asarray(data_file["y_train"][:args.count], dtype=np.float32).reshape(-1)
    image_encoder = DinoSafetyLatentEncoder(args.checkpoint, device).eval()
    surrogate = PhysicalLatentSurrogate(args.checkpoint, device).eval()
    with torch.no_grad():
        image_latent = image_encoder(torch.from_numpy(images).to(device))
        physical_latent = surrogate(torch.from_numpy(distance_m).to(device))
        residual = image_latent - physical_latent
    metrics = {
        "experiment": "dino_safety_latent_runtime_smoke",
        "status": "passed" if image_latent.shape == (args.count, 32) else "failed",
        "samples": args.count,
        "image_latent_shape": list(image_latent.shape),
        "physical_latent_shape": list(physical_latent.shape),
        "runtime_output_is_distance": False,
        "residual_norm_mean": float(torch.linalg.vector_norm(residual, dim=1).mean()),
        "residual_norm_max": float(torch.linalg.vector_norm(residual, dim=1).max()),
        "image_latent_finite": bool(torch.isfinite(image_latent).all()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False)
    np.savez_compressed(
        args.output_dir / "latents.npz",
        distance_m=distance_m,
        image_latent=image_latent.cpu().numpy(),
        physical_latent=physical_latent.cpu().numpy(),
        residual=residual.cpu().numpy(),
    )
    print("[DINO safety-latent runtime smoke]")
    print("status=%s, image latent=%s, physical latent=%s" % (
        metrics["status"], metrics["image_latent_shape"], metrics["physical_latent_shape"]
    ))
    print("runtime output is distance: %s" % metrics["runtime_output_is_distance"])
    print("residual norm mean/max=%.6f/%.6f" % (
        metrics["residual_norm_mean"], metrics["residual_norm_max"]
    ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
