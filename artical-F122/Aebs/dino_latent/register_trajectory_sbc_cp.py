"""Freeze the trajectory-SBC CP experiment before collecting fresh data."""

import argparse
import json
import time
from pathlib import Path

import torch

from Aebs.dino_latent.prepare_expanded_data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--barrier", type=Path,
        default=Path(
            "results/dino_expanded_v1/19_trajectory_scenario_sbc_boundary_v3/barrier.pt"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/20_trajectory_sbc_cp_registration_v1"),
    )
    parser.add_argument("--collection-seed", type=int, default=91009)
    parser.add_argument("--episodes", type=int, default=459)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.01)
    args = parser.parse_args()
    if (
        args.output_dir.exists() or not args.barrier.is_file()
        or args.episodes < 459 or args.horizon < 1
        or not 0.0 < args.epsilon < 1.0 or not 0.0 < args.beta < 1.0
    ):
        parser.error("missing barrier, existing output, or invalid settings")
    checkpoint = torch.load(args.barrier, map_location="cpu")
    if (
        checkpoint.get("experiment")
        != "R2_trajectory_scenario_SBC_boundary_focused_v3"
        or checkpoint.get("development_only") is not True
    ):
        parser.error("unexpected barrier checkpoint")
    registration = {
        "experiment": "R2_frozen_trajectory_SBC_conformal_safety_v1",
        "registered_at_unix": time.time(),
        "barrier": str(args.barrier),
        "barrier_sha256": sha256(args.barrier),
        "barrier_training_h5_sha256": checkpoint["trajectory_h5_sha256"],
        "representation_sha256": checkpoint["representation_sha256"],
        "controller_sha256": checkpoint["controller_sha256"],
        "barrier_frozen": True,
        "models_frozen": True,
        "collection": {
            "protocol": "original_spvc_primary_monitor_fullscreen_v1",
            "role": "calibration",
            "seed": args.collection_seed,
            "episodes": args.episodes,
            "horizon": args.horizon,
        },
        "score": "B(s0) + sum_t max(B(s_next)-B(s),0) - 1",
        "stopped_boundary": {"safe_terminal": 0.0, "unsafe": 1.0},
        "epsilon": args.epsilon,
        "beta": args.beta,
        "pass_condition": "trajectory-level conformal q_hat < 0",
        "test_used": False,
    }
    args.output_dir.mkdir(parents=True)
    path = args.output_dir / "registration.json"
    path.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    print("[R2 trajectory-SBC CP preregistration]")
    print("barrier_sha256:", registration["barrier_sha256"])
    print("collection:", registration["collection"])
    print("epsilon=%.6f beta=%.6f" % (args.epsilon, args.beta))
    print("registration:", path)


if __name__ == "__main__":
    main()
