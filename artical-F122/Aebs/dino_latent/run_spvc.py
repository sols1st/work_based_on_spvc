"""Run original SPVC with only its cGAN/state-net front end replaced by R2 DINO.

No conformal contract, residual box, recoverability calculation, QP, safety
filter, or replacement verifier is used.  The physical grid is mapped through
the frozen q_psi branch to the same latent coordinates used by the frozen-DINO
image branch; the original SPVC learner/verifier then trains the PPO actor and
SBC exactly through their legacy interfaces.
"""

import argparse
import json
from pathlib import Path

import torch

from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.compare_spvc_scopes import compare_scopes
from Aebs.system.env import Aebs
from Aebs.VT.loop import Loop
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation-checkpoint",
        type=Path,
        default=Path(
            "results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"
        ),
    )
    parser.add_argument(
        "--controller",
        type=Path,
        default=Path("results/dino_expanded_v1/09_R2_ppo/latent_ppo.zip"),
    )
    parser.add_argument(
        "--controller-metrics", type=Path,
        default=Path("results/dino_expanded_v1/09_R2_ppo/metrics.json"),
    )
    parser.add_argument(
        "--initial-sbc", type=Path, default=None,
        help="Optional same-method warm start. Omit for the clean front-end-only baseline.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dino_expanded_v1/17_R2_original_spvc"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-iteration-index", type=int, default=100)
    parser.add_argument("--stop-on-zero-violation", action="store_true")
    parser.add_argument("--compare-scopes", action="store_true",
                        help="Post-run legacy narrow/broad checks on the same saved models; no training changes.")
    parser.add_argument("--diagnose-only-from", type=Path, default=None,
                        help="Load a previous run_spvc output; skip training and compare scopes in a NEW output dir.")
    args = parser.parse_args()

    required = [
        args.representation_checkpoint, args.controller, args.controller_metrics,
    ]
    if args.initial_sbc is not None:
        required.append(args.initial_sbc)
    if args.diagnose_only_from is not None:
        required.extend(args.diagnose_only_from / name for name in (
            "sbc.pt", "latent_ppo_spvc.pt", "metrics.json"
        ))
        if args.initial_sbc is not None:
            parser.error("--initial-sbc cannot be combined with --diagnose-only-from")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing inputs: " + ", ".join(missing))
    if args.timeout_seconds <= 0 or args.max_iteration_index < 0:
        parser.error("timeout must be positive and iteration index nonnegative")

    representation = torch.load(args.representation_checkpoint, map_location="cpu")
    controller_metrics = json.loads(args.controller_metrics.read_text(encoding="utf-8"))
    if (
        representation.get("variant") != "R2_group_alignment"
        or representation.get("dino_frozen") is not True
        or controller_metrics.get("representation_sha256")
        != sha256(args.representation_checkpoint)
        or controller_metrics.get("checkpoint_sha256") != sha256(args.controller)
        or controller_metrics.get("test_used") is not False
    ):
        parser.error("R2 representation/PPO provenance mismatch")

    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("output directory is nonempty; choose a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DinoLatentSPVCPolicy.from_checkpoints(
        args.representation_checkpoint, args.controller, device
    )
    env = Aebs(0.05)
    policy.set_state_distance_scale(env.std1)
    source_metrics = None
    if args.diagnose_only_from is not None:
        source_metrics = json.loads((args.diagnose_only_from / "metrics.json").read_text(encoding="utf-8"))
        if (source_metrics.get("representation_sha256") != sha256(args.representation_checkpoint)
                or source_metrics.get("controller_sha256") != sha256(args.controller)):
            parser.error("diagnostic source does not match representation/base PPO")
        # Preserve the original run's input interpretation, even for pre-fix runs.
        old_policy = source_metrics.get("policy", {})
        policy.set_state_distance_scale(old_policy.get("state_distance_scale_m", policy.distance_scale_m))
        policy.load_state_dict(torch.load(args.diagnose_only_from / "latent_ppo_spvc.pt", map_location=device))
    learner = VTLearner(
        l_model_config=[2, 16, 8, 1],
        env=env,
        p_lip=2.0,
        l_lip=4.0,
        eps=0.1,
        gamma_decrease=1.0,
        reach_prob=0.95,
        square_l_output=True,
        l_model_path=str(args.diagnose_only_from / "sbc.pt") if args.diagnose_only_from is not None
        else (str(args.initial_sbc) if args.initial_sbc is not None else None),
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
    if args.diagnose_only_from is None:
        loop.run(args.timeout_seconds)
    else:
        loop.info = {"diagnostic_only": True, "source_loop_info": source_metrics.get("loop_info")}
    torch.save(learner.l_model.state_dict(), output_dir / "sbc.pt")
    torch.save(learner.p_net.state_dict(), output_dir / "latent_ppo_spvc.pt")
    result = {
        "experiment": "R2_DINO_original_SPVC_front_end_only_v1",
        "scope": "original SPVC method with only cGAN/state_net replaced by R2 DINO latent",
        "claim_boundary": (
            "legacy SPVC assumptions and verifier only; this run does not add a formal "
            "bound for image-latent versus q-latent mismatch"
        ),
        "representation_checkpoint": str(args.representation_checkpoint),
        "representation_sha256": sha256(args.representation_checkpoint),
        "controller_checkpoint": str(args.controller),
        "controller_sha256": sha256(args.controller),
        "controller_metrics": str(args.controller_metrics),
        "initial_sbc": str(args.initial_sbc) if args.initial_sbc is not None else None,
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
            "noise_model": "original uniform next-state noise with factor=0.05",
            "interval_implementation": "legacy VTVerifier unchanged",
        },
        "excluded": [
            "conformal prediction",
            "global or conditional residual contract",
            "recoverable set",
            "QP",
            "safety filter",
            "new verifier",
        ],
        "test_used": False,
        "training_run": args.diagnose_only_from is None,
        "diagnose_only_from": str(args.diagnose_only_from) if args.diagnose_only_from else None,
        "saved_barrier_sha256": sha256(output_dir / "sbc.pt"),
        "saved_policy_sha256": sha256(output_dir / "latent_ppo_spvc.pt"),
        "scope_comparison_requested": args.compare_scopes or args.diagnose_only_from is not None,
        "loop_info": loop.info,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False, default=str)
    print("\n[R2 DINO front end + original SPVC]")
    print("front end: image -> frozen DINO/projection -> latent; VT proxy: distance -> q_psi latent")
    print("removed: cGAN, legacy state_net")
    print("PPO/SBC/noise/grid/VT settings: original SPVC")
    print("CP/residual box/QP/new verifier: not used")
    print("claim: legacy SPVC baseline; DINO dual-path mismatch is not formally covered")
    print("hard violations:", loop.info.get("hard_violations"))
    print("init upper:", loop.info.get("ub_init"))
    print("unsafe lower:", loop.info.get("lb_unsafe"))
    print("domain lower:", loop.info.get("domain_min"))
    print("safe-reach probability lower bound:", loop.info.get("actual_reach_prob"))
    print("checkpoint matches verification:", loop.info.get("checkpoint_matches_verification", False))
    print("result directory: %s" % output_dir)
    if args.compare_scopes or args.diagnose_only_from is not None:
        # Build a fresh bound wrapper for exactly the final saved model, not the
        # last iteration's possibly stale pre-PPO-update diagnostics.
        diagnostic_verifier = VTVerifier(
            learner, env, learner.create_bounded_module(learner.l_model),
            batch_size=2048, reach_prob=0.9, fail_check_fast=False,
        )
        denominator = len(verifier.train_buffer)
        if source_metrics is not None:
            denominator = int(source_metrics.get("loop_info", {}).get("ds_size", 10000))
        comparison = compare_scopes(diagnostic_verifier, output_dir, denominator)
        comparison["saved_barrier_sha256"] = result["saved_barrier_sha256"]
        comparison["saved_policy_sha256"] = result["saved_policy_sha256"]
        comparison["policy"] = policy.metadata()
        with (output_dir / "scope_comparison.json").open("w", encoding="utf-8") as stream:
            json.dump(comparison, stream, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == "__main__":
    main()
