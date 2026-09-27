"""Run original SPVC training with only the observation front end replaced.

This entry point deliberately reuses ``Aebs.VT.VTLearner``, ``VTVerifier``,
``Loop``, ``Aebs`` and all their original hyperparameters.  It does not load
the cGAN, state_net, safety filter, conformal contract, recoverable-set code,
or any of the later MVP verifiers.
"""

import argparse
import json
from pathlib import Path

import torch

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.system.env import Aebs
from Aebs.VT.loop import Loop
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier


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
    parser.add_argument(
        "--output-dir",
        default="results/semantic_spvc_minimal",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-iteration-index", type=int, default=100)
    parser.add_argument("--stop-on-zero-violation", action="store_true")
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args()

    resume_metrics = None
    if args.resume_dir is not None:
        with open(args.resume_dir / "metrics.json", encoding="utf-8") as metrics_file:
            resume_metrics = json.load(metrics_file)
        start_iteration = int(resume_metrics["loop_info"]["iter"]) + 1
        if args.max_iteration_index < start_iteration:
            parser.error("--max-iteration-index must be at least the next iteration")
    else:
        start_iteration = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    semantic_policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint,
        args.controller,
        device,
    )

    # These values are copied from the original Aebs.VT.loop entry point.
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
        l_model_path=str(args.resume_dir / "sbc.pt") if args.resume_dir else None,
        p_model_path=str(args.resume_dir / "semantic_ppo.pt") if args.resume_dir else None,
        p_net=semantic_policy,
    )
    l_ibp = learner.create_bounded_module(learner.l_model)
    verifier = VTVerifier(
        learner,
        env,
        l_ibp,
        batch_size=2048,
        reach_prob=0.9,
        fail_check_fast=True,
    )
    loop = Loop(
        learner,
        verifier,
        env,
        jitter_grid=env.space_split,
        soft_constraint=False,
        max_iteration_index=args.max_iteration_index,
        stop_on_zero_violation=args.stop_on_zero_violation,
    )
    loop.iter = start_iteration
    loop.run(args.timeout_seconds)

    torch.save(learner.l_model.state_dict(), output_dir / "sbc.pt")
    torch.save(learner.p_net.state_dict(), output_dir / "semantic_ppo.pt")
    result = {
        "experiment": "spvc_semantic_only",
        "semantic_checkpoint": args.semantic_checkpoint,
        "controller_checkpoint": args.controller,
        "resume_dir": str(args.resume_dir) if args.resume_dir else None,
        "start_iteration": start_iteration,
        "max_iteration_index": args.max_iteration_index,
        "stop_on_zero_violation": args.stop_on_zero_violation,
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
            "batch_size": 2048,
            "fail_check_fast": True,
        },
        "loop_info": loop.info,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2, ensure_ascii=False, default=str)

    print("\n[Semantic-only SPVC]")
    print("front end: real image -> semantic encoder -> original PPO")
    print("cGAN: removed")
    print("legacy state_net: removed")
    print("PPO/SBC/VT verification settings: unchanged")
    print(f"result directory: {output_dir}")


if __name__ == "__main__":
    main()
