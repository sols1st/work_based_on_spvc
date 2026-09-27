"""Read-only recheck of a saved semantic-SPVC PPO/SBC pair with fixed IBP.

This does not continue training, overwrite old metrics, or establish a formal
certificate by itself. It repairs both state-set and disturbance-cell interval
calls, then reuses the original VT grid filter and decrease test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.semantic_spvc.corrected_verify import CorrectedIntervalVTVerifier
from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_corrected_recheck"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-bins", type=int, default=10,
                        help="Uniform disturbance-box cells per dimension; support is unchanged")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 1 <= args.noise_bins <= 100:
        parser.error("--noise-bins must be between 1 and 100")
    if args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("--output-dir must differ from --checkpoint-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint, args.controller, device
    )
    policy.load_state_dict(torch.load(
        args.checkpoint_dir / "semantic_ppo.pt", map_location=device
    ))
    policy.eval()
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    barrier.eval()
    for model in (policy, barrier):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    from types import SimpleNamespace
    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = CorrectedIntervalVTVerifier(
        SimpleNamespace(p_net=policy, l_model=barrier), env, bounded,
        batch_size=args.batch_size, reach_prob=0.9, fail_check_fast=False,
    )
    verifier.pmass_n = np.full(env.observation_space.shape[0], args.noise_bins, dtype=int)
    verifier._cached_pmass_grid = None
    # Original loop calls check_dec_cond(1.2). The repaired subclass changes
    # interval inputs only; filtering and hard violation definition stay intact.
    _, original_threshold_count, check_info, _ = verifier.check_dec_cond(1.2)
    _, init_upper = verifier.compute_bound_init(env.space_split)
    unsafe_lower, _ = verifier.compute_bound_unsafe(env.space_split)
    domain_lower, _ = verifier.compute_bound_domain(env.space_split)

    grid, _, _ = verifier.get_unfiltered_grid(env.space_split)
    values = []
    with torch.no_grad():
        for start in range(0, len(grid), args.batch_size):
            batch = torch.as_tensor(grid[start:start + args.batch_size],
                                    dtype=torch.float32, device=device)
            values.append(barrier(batch).flatten().cpu().numpy())
    values = np.concatenate(values)
    region_low = domain_lower + 0.95 * (unsafe_lower - domain_lower)
    region_count = int(np.sum((values > region_low) & (values < unsafe_lower)))

    violations = int(original_threshold_count)
    candidate_probability = None
    if violations == 0 and unsafe_lower > init_upper and init_upper > domain_lower:
        # Same algebra as the original loop, reported only as an arithmetic
        # candidate: other proof assumptions are not audited here.
        candidate_probability = 1.0 - (init_upper - domain_lower) / (unsafe_lower - domain_lower)

    metrics = {
        "experiment": "semantic_spvc_corrected_interval_recheck",
        "level": "corrected_legacy_grid_check_not_a_full_safety_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "grid_states": int(np.prod(env.space_split)),
        "noise_bins_per_dimension": args.noise_bins,
        "noise_cells": args.noise_bins ** env.observation_space.shape[0],
        "original_filtered_region_states": region_count,
        "corrected_ibp_hard_violations": violations,
        "original_threshold_at_most_0_1_percent": bool(
            violations / int(np.prod(env.space_split)) <= 0.001
        ),
        "corrected_region_bounds": {
            "init_upper": init_upper,
            "unsafe_lower": unsafe_lower,
            "domain_lower": domain_lower,
            "filter_lower_exclusive": region_low,
        },
        "candidate_probability_if_zero_violations": candidate_probability,
        "verifier_info": check_info,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, default=str)

    print("[Semantic-SPVC corrected interval recheck: saved models, no training]")
    print("grid=%d, noise cells=%d, corrected filter region=%d, hard violations=%d" % (
        metrics["grid_states"], metrics["noise_cells"], region_count, violations
    ))
    print("bounds: init upper=%.6f, unsafe lower=%.6f, domain lower=%.6f" % (
        init_upper, unsafe_lower, domain_lower
    ))
    print("original <=0.1%% threshold=%s, zero violations=%s" % (
        metrics["original_threshold_at_most_0_1_percent"], violations == 0
    ))
    print("candidate probability (only if zero violations)=%s" % candidate_probability)
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
