"""Fast interface checks for the semantic-only SPVC variant."""

import argparse

import h5py
import numpy as np
import torch

from Aebs.semantic_spvc.model import SemanticSPVCPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-checkpoint",
        default="results/mvp/01_semantic/semantic_encoder.pt",
    )
    parser.add_argument(
        "--controller",
        default="Aebs/controller/best_model/best_model.zip",
    )
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint,
        args.controller,
        device,
    ).eval()

    with h5py.File(args.data, "r") as data_file:
        images = torch.from_numpy(np.asarray(data_file["X_train"][:8], dtype=np.float32)).to(device)
        distance_m = np.asarray(data_file["y_train"][:8], dtype=np.float32).reshape(-1)
    speed = torch.linspace(0.0, 3.0, len(images), device=device)
    true_state = torch.from_numpy(
        np.stack([distance_m / policy.distance_scale_m, speed.cpu().numpy()], axis=1).astype(np.float32)
    ).to(device)
    z = torch.zeros(len(images), 4, device=device)

    with torch.no_grad():
        semantic_mean, semantic_scale = policy.encode_image(images)
        image_action = policy.forward_image(images, speed)
        state_action = policy(z, true_state)

    assert semantic_mean.shape == (len(images), 1)
    assert semantic_scale.shape == (len(images), 1)
    assert image_action.shape == (len(images), 1)
    assert state_action.shape == (len(images), 1)
    assert torch.isfinite(semantic_mean).all()
    assert torch.isfinite(semantic_scale).all()
    assert torch.isfinite(image_action).all()
    assert torch.isfinite(state_action).all()

    print("[Semantic-only SPVC smoke test]")
    print(f"samples: {len(images)}")
    print(f"semantic mean range: {semantic_mean.min().item():.6f}..{semantic_mean.max().item():.6f}")
    print(f"semantic scale range: {semantic_scale.min().item():.6f}..{semantic_scale.max().item():.6f}")
    print(f"image action range: {image_action.min().item():.6f}..{image_action.max().item():.6f}")
    print(f"state action range: {state_action.min().item():.6f}..{state_action.max().item():.6f}")
    print("status: passed")


if __name__ == "__main__":
    main()
