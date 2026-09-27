"""Minimal semantic-SPVC alternating update with a differentiable QP output.

The original VTLearner losses, optimizers, dynamics and VTVerifier are reused.
The optional small state subset is solely for a bounded end-to-end smoke run.
Saved base-policy and SBC checkpoints remain compatible with existing tools.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_policy import SemanticSPVCQPPolicy
from Aebs.system.env import Aebs
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-dir", type=Path, default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_joint_smoke"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--train-states", type=int, default=512,
                        help="0 uses the full original 100x100 training grid")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--l-epochs", type=int, default=1)
    parser.add_argument("--p-epochs", type=int, default=1)
    parser.add_argument("--slack-weight", type=float, default=100.0)
    parser.add_argument("--p-learning-rate", type=float, default=None,
                        help="Optional actor LR for QP fine-tuning; default keeps original SPVC 5e-2")
    parser.add_argument("--original-controller-init", action="store_true",
                        help="Resume SBC only; initialize actor/teacher from the original pretrained PPO")
    args = parser.parse_args()
    if args.batch_size < 1 or args.l_epochs < 1 or args.p_epochs < 1:
        parser.error("Batch size and epoch counts must be positive")
    if args.p_learning_rate is not None and args.p_learning_rate <= 0:
        parser.error("--p-learning-rate must be positive")
    if not 0 <= args.train_states <= 10000:
        parser.error("--train-states must be in [0, 10000]")
    if args.output_dir.resolve() == args.resume_dir.resolve():
        parser.error("Output directory must differ from resume directory")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is nonempty; use a new directory to preserve results")

    started = time.time()
    env = Aebs(0.05)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = SemanticSPVCPolicy.from_checkpoints(args.semantic_checkpoint, args.controller, device)
    learner = VTLearner(
        l_model_config=[2, 16, 8, 1], env=env, p_lip=2.0, l_lip=4.0,
        eps=0.1, gamma_decrease=1.0, reach_prob=0.95, square_l_output=True,
        l_model_path=str(args.resume_dir / "sbc.pt"),
        p_model_path=(None if args.original_controller_init else
                      str(args.resume_dir / "semantic_ppo.pt")), p_net=base,
    )
    # VTLearner copied its frozen teacher and created the actor optimizer before
    # attaching QP. Thus the original teacher stays nominal and the optimizer
    # still owns exactly the original controller parameters.
    learner.p_net = SemanticSPVCQPPolicy(base, learner.l_model, env, args.slack_weight).to(device)
    actor_ids = {id(parameter) for group in learner.p_optimizer.param_groups
                 for parameter in group["params"]}
    assert actor_ids == {id(parameter) for parameter in base.controller_net.parameters()}
    assert not actor_ids.intersection({id(parameter) for parameter in learner.l_model.parameters()})
    if args.p_learning_rate is not None:
        for group in learner.p_optimizer.param_groups:
            group["lr"] = args.p_learning_rate

    # The original create_bounded_module() freezes the SBC parameters. Here the
    # wrapper is created only after both alternating updates, so the SBC really
    # participates in the requested training and verification uses final weights.
    from Aebs.VT.verify import VTVerifier as GridSource
    source = GridSource(SimpleNamespace(), env, None, batch_size=64,
                        reach_prob=0.9, fail_check_fast=False)
    grid, _, _ = source.get_unfiltered_grid(env.train_space_split)
    if args.train_states:
        selected = np.linspace(0, len(grid) - 1, args.train_states, dtype=np.int64)
        grid = grid[selected]
    delta = (env.observation_space.high - env.observation_space.low) / env.train_space_split
    print("[QP joint training] states=%d, batch=%d, l_epochs=%d, p_epochs=%d" % (
        len(grid), args.batch_size, args.l_epochs, args.p_epochs
    ), flush=True)
    l_history = learner.train_epoch(
        grid, delta, lip=0.001, batch_size=args.batch_size, shuffle=True,
        num_epochs=args.l_epochs, train_fn="l",
    )
    print("SBC training finished: %s" % l_history[-1], flush=True)
    p_history = learner.train_epoch(
        grid, delta, lip=0.001, batch_size=args.batch_size, shuffle=True,
        num_epochs=args.p_epochs, train_fn="p",
    )
    print("PPO+QP training finished: %s" % p_history[-1], flush=True)

    # Original verifier, same grid/noise/threshold; it now sees the final QP
    # action rather than the nominal PPO action. Its legacy interval limitation
    # is recorded below, not silently treated as a formal guarantee.
    bounded = learner.create_bounded_module(learner.l_model)
    verifier = VTVerifier(learner, env, bounded, batch_size=128,
                          reach_prob=0.9, fail_check_fast=False)
    verifier.prefill_train_buffer()
    _, hard_count, verify_info, _ = verifier.check_dec_cond(1.2)
    _, init_upper = verifier.compute_bound_init(env.space_split)
    unsafe_lower, _ = verifier.compute_bound_unsafe(env.space_split)
    domain_lower, _ = verifier.compute_bound_domain(env.space_split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(learner.l_model.state_dict(), args.output_dir / "sbc.pt")
    torch.save(base.state_dict(), args.output_dir / "semantic_ppo.pt")
    metrics = {
        "experiment": "semantic_spvc_qp_joint_smoke",
        "scope": "original_spvc_losses_dynamics_and_legacy_verifier_with_qp_output",
        "resume_dir": str(args.resume_dir),
        "controller_initialization": ("original_pretrained_ppo" if args.original_controller_init
                                      else str(args.resume_dir / "semantic_ppo.pt")),
        "training": {
            "states": len(grid), "batch_size": args.batch_size,
            "l_epochs": args.l_epochs, "p_epochs": args.p_epochs,
            "p_learning_rate": learner.p_optimizer.param_groups[0]["lr"],
            "l_last": l_history[-1], "p_last": p_history[-1],
            "elapsed_seconds": time.time() - started,
        },
        "legacy_spvc_verification": {
            "hard_violations": int(hard_count), "denominator": 10000,
            "original_pass_threshold": bool(int(hard_count) / 10000 <= 0.001),
            "init_upper": init_upper, "unsafe_lower": unsafe_lower,
            "domain_lower": domain_lower, "info": verify_info,
            "warning": "Original VT interval call is not a valid IBP upper bound; these are legacy experimental values, not a safety certificate.",
        },
        "policy": learner.p_net.metadata(),
        "checkpoint_files": ["semantic_ppo.pt", "sbc.pt"],
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, default=str)
    print("[QP joint training completed] legacy violations=%d/10000, elapsed=%.1fs" % (
        int(hard_count), metrics["training"]["elapsed_seconds"]
    ))
    print("result directory: %s" % args.output_dir)


if __name__ == "__main__":
    main()
