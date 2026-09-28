"""Split-conformal calibration of the 32-D DINO safety-latent residual."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.dino_latent.models import PhysicalToLatent


def split_conformal_radius(scores, alpha):
    count = len(scores)
    rank = int(math.ceil((count + 1) * (1.0 - alpha)))
    if rank > count:
        raise ValueError("calibration sample count is too small for requested alpha")
    return float(np.sort(scores)[rank - 1]), rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation-dir", type=Path,
                        default=Path("results/dino_safety_latent_stage1"))
    parser.add_argument("--data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dino_latent_cp_stage2"))
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie in (0,1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")

    checkpoint_path = args.representation_dir / "dino_safety_latent.pt"
    latent_path = args.representation_dir / "latent_data.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved = np.load(latent_path)
    latent = torch.from_numpy(saved["latent"]).float()
    calibration_idx = saved["split_calibration"].astype(np.int64)
    test_idx = saved["split_test"].astype(np.int64)
    with h5py.File(args.data, "r") as data_file:
        distance_m = np.asarray(data_file["y_train"], dtype=np.float32).reshape(-1)

    q_model = PhysicalToLatent(
        physical_dim=1, latent_dim=int(checkpoint["latent_dimension"])
    )
    q_model.load_state_dict(checkpoint["physical_to_latent_state_dict"])
    q_model.eval()
    normalized_distance = torch.from_numpy(
        distance_m / float(checkpoint["distance_scale_m"])
    ).reshape(-1, 1)
    with torch.no_grad():
        center = q_model(normalized_distance)
    residual = latent - center
    scores = torch.linalg.vector_norm(residual, dim=1).numpy()
    radius, rank = split_conformal_radius(scores[calibration_idx], args.alpha)
    test_scores = scores[test_idx]
    covered = test_scores <= radius

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": "dino_32d_latent_split_conformal",
        "representation_checkpoint": str(checkpoint_path),
        "latent_dimension": int(checkpoint["latent_dimension"]),
        "nonconformity_score": "L2 norm of z_DINO_projection - q_psi(distance)",
        "set": "Z_alpha(d) = {z: ||z-q_psi(d)||_2 <= radius}",
        "alpha": args.alpha,
        "target_marginal_coverage": 1.0 - args.alpha,
        "calibration_count": int(len(calibration_idx)),
        "finite_sample_rank": rank,
        "radius": radius,
        "calibration_score": {
            "mean": float(scores[calibration_idx].mean()),
            "median": float(np.median(scores[calibration_idx])),
            "max": float(scores[calibration_idx].max()),
        },
        "independent_test": {
            "count": int(len(test_idx)),
            "covered": int(covered.sum()),
            "coverage": float(covered.mean()),
            "mean_score": float(test_scores.mean()),
            "max_score": float(test_scores.max()),
            "excess_count": int((~covered).sum()),
            "maximum_excess": float(np.maximum(test_scores - radius, 0.0).max()),
        },
        "claim_boundary": (
            "Marginal latent coverage under exchangeability of the current image dataset; "
            "not conditional coverage, not controller safety, and not a formal closed-loop certificate"
        ),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(
        args.output_dir / "latent_cp_data.npz",
        calibration_indices=calibration_idx,
        test_indices=test_idx,
        calibration_scores=scores[calibration_idx],
        test_scores=test_scores,
        radius=np.array(radius, dtype=np.float32),
    )
    print("[DINO 32-D latent split conformal]")
    print("calibration=%d, alpha=%.3f, rank=%d, radius=%.6f" % (
        len(calibration_idx), args.alpha, rank, radius
    ))
    print("independent test coverage=%d/%d = %.2f%%, excess=%d, max excess=%.6f" % (
        covered.sum(), len(test_idx), covered.mean() * 100,
        (~covered).sum(), metrics["independent_test"]["maximum_excess"],
    ))
    print("scope: latent marginal coverage only; not a controller-safety certificate")
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
