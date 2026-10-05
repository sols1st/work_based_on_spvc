"""Run the original SPVC SBC loop with the no-CP DINO latent PPO."""

import argparse
import json
from pathlib import Path

import torch

from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
from Aebs.system.env import Aebs
from Aebs.VT.loop import Loop
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation-checkpoint",
        default="results/dino_safety_latent_stage1/dino_safety_latent.pt",
    )
    parser.add_argument(
        "--controller",
        default="results/dino_latent_ppo_stage2_mixed_20260928/latent_ppo.zip",
    )
    parser.add_argument("--initial-sbc", default="results/semantic_spvc_my_run/sbc.pt")
    parser.add_argument("--output-dir", default="results/dino_latent_spvc_stage3")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-iteration-index", type=int, default=500)
    parser.add_argument("--stop-on-zero-violation", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DinoLatentSPVCPolicy.from_checkpoints(
        args.representation_checkpoint, args.controller, device
    )
    env = Aebs(0.05)
    learner = VTLearner(
        l_model_config=[2, 16, 8, 1],
        env=env,
        p_lip=2.0,
        l_lip=4.0,
        eps=0.1,
        gamma_decrease=1.0,
        reach_prob=0.95,
        square_l_output=True,
        l_model_path=args.initial_sbc,
        p_net=policy,
    )
    l_ibp = learner.create_bounded_module(learner.l_model)
    verifier = VTVerifier(
        learner, env, l_ibp,
        batch_size=2048, reach_prob=0.9, fail_check_fast=True,
    )
    loop = Loop(
        learner, verifier, env,
        jitter_grid=env.space_split,
        soft_constraint=False,
        max_iteration_index=args.max_iteration_index,
        stop_on_zero_violation=args.stop_on_zero_violation,
    )
    loop.run(args.timeout_seconds)
    torch.save(learner.l_model.state_dict(), output_dir / "sbc.pt")
    torch.save(learner.p_net.state_dict(), output_dir / "latent_ppo_spvc.pt")
    result = {
        "experiment": "dino_latent_spvc_stage3_no_cp",
        "representation_checkpoint": args.representation_checkpoint,
        "controller_checkpoint": args.controller,
        "initial_sbc": args.initial_sbc,
        "policy": learner.p_net.metadata(),
        "original_vt_settings": {
            "environment_factor": 0.05,
            "l_model_config": [2, 16, 8, 1],
            "p_lip": 2.0,
            "l_lip": 4.0,
            "epsilon": 0.1,
            "gamma_decrease": 1.0,
            "learner_reach_prob": 0.95,
            "verifier_reach_prob": 0.9,
            "grid": [100, 100],
            "hard_violation_fraction_threshold": 0.001,
        },
        "loop_info": loop.info,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False, default=str)
    print("\n[DINO latent SPVC: no CP]")
    print("PPO: 32-D latent + speed")
    print("SBC/VT settings: original SPVC")
    print("result directory: %s" % output_dir)


if __name__ == "__main__":
    main()
