"""Check the saved image->semantic->PPO->SBC-QP inference path."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_policy import SemanticSPVCQPPolicy
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_joint_full_20260926"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--data", default="Aebs/data/Downsampled.h5")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_image_smoke"))
    args = parser.parse_args()
    if args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("--output-dir must differ from --checkpoint-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    base = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint, args.controller, device
    )
    base.load_state_dict(torch.load(args.checkpoint_dir / "semantic_ppo.pt", map_location=device))
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    policy = SemanticSPVCQPPolicy(base, barrier, env).to(device).eval()
    with h5py.File(args.data, "r") as data_file:
        images = torch.from_numpy(np.asarray(data_file["X_train"][:8], dtype=np.float32)).to(device)
    speed = torch.linspace(0.0, 3.0, len(images), device=device)
    with torch.no_grad():
        semantic_mean, semantic_scale = base.encode_image(images)
        action = policy.forward_image(images, speed)
    if action.shape != (len(images), 1) or not torch.isfinite(action).all():
        raise RuntimeError("Image->QP action is malformed or nonfinite")
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
    if bool(((action < low - 1e-6) | (action > high + 1e-6)).any()):
        raise RuntimeError("QP output is outside the action bounds")
    metrics = {
        "experiment": "semantic_spvc_qp_image_smoke",
        "level": "forward_interface_check_not_a_closed_loop_or_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "images": int(len(images)),
        "semantic_mean_range": [float(semantic_mean.min()), float(semantic_mean.max())],
        "semantic_scale_range": [float(semantic_scale.min()), float(semantic_scale.max())],
        "qp_action_range": [float(action.min()), float(action.max())],
        "status": "passed",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    print("[Semantic-SPVC QP full image-path smoke] status=passed, images=%d, "
          "action range=[%.6f, %.6f]" % (
              len(images), metrics["qp_action_range"][0], metrics["qp_action_range"][1]
          ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
