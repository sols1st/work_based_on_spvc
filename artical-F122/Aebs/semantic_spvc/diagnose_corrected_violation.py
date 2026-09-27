"""Locate corrected-grid counterexamples and test the frozen QP at them.

Only existing PPO/SBC checkpoints are read. A finite action scan is diagnostic,
not a proof that a feasible action does or does not exist on the continuum.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.semantic_spvc.corrected_verify import CorrectedIntervalVTVerifier
from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def corrected_residual(verifier, barrier, states, actions, batch_size):
    pmass, noise_low, noise_high = verifier.get_pmass_grid()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            s = states[start:start + batch_size]
            a = actions[start:start + batch_size]
            expected_upper = verifier.compute_expected_l(s, a, pmass, noise_low, noise_high)
            pieces.append(expected_upper - barrier(s).flatten())
    return torch.cat(pieces).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--recheck-dir", type=Path,
                        default=Path("results/semantic_spvc_corrected_recheck"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_corrected_violation_diagnostic"))
    parser.add_argument("--action-points", type=int, default=121)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.action_points < 3 or args.batch_size < 1:
        parser.error("--action-points must be >= 3 and --batch-size must be positive")
    if args.output_dir.resolve() in (args.checkpoint_dir.resolve(), args.recheck_dir.resolve()):
        parser.error("--output-dir must differ from input directories")

    with open(args.recheck_dir / "metrics.json", encoding="utf-8") as input_file:
        recheck = json.load(input_file)
    if Path(recheck["checkpoint_dir"]).resolve() != args.checkpoint_dir.resolve():
        raise ValueError("Recheck metrics refer to a different checkpoint directory")
    bounds = recheck["corrected_region_bounds"]
    unsafe_lower = float(bounds["unsafe_lower"])
    region_lower = float(bounds["filter_lower_exclusive"])
    if not np.isfinite(unsafe_lower) or not np.isfinite(region_lower):
        raise ValueError("Invalid saved corrected region bounds")

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
    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = CorrectedIntervalVTVerifier(
        SimpleNamespace(p_net=policy, l_model=barrier), env, bounded,
        batch_size=args.batch_size, reach_prob=0.9, fail_check_fast=False,
    )

    total = int(np.prod(env.space_split))
    # Match the exact grid-center generator used by check_dec_cond(), rather
    # than the slightly different NumPy grid used in the Q1c convenience path.
    full_grid = verifier.v_get_grid_item(torch.arange(total)).to(device)
    with torch.no_grad():
        values = barrier(full_grid).flatten()
    mask = (values > region_lower) & (values < unsafe_lower)
    states = full_grid[mask]
    if len(states) == 0:
        raise RuntimeError("Corrected verifier region has no grid centers")
    with torch.no_grad():
        latent = torch.zeros((len(states), 4), dtype=states.dtype, device=device)
        nominal = policy(latent, states)
    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    affine = constraint.linearize(states, nominal)
    qp = ScalarSBCQP(
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]), slack_weight=100.0,
    ).to(device)
    solution = qp(nominal, affine.coefficient, affine.right_hand_side)
    nominal_residual = corrected_residual(
        verifier, barrier, states, nominal, args.batch_size
    )
    qp_residual = corrected_residual(
        verifier, barrier, states, solution.action, args.batch_size
    )
    midpoint_residual = constraint.residual(states, solution.action).detach().cpu().numpy().reshape(-1)
    states_np = states.detach().cpu().numpy()
    nominal_np = nominal.detach().cpu().numpy().reshape(-1)
    qp_np = solution.action.detach().cpu().numpy().reshape(-1)
    slack_np = solution.slack.detach().cpu().numpy().reshape(-1)

    cases = []
    action_grid = torch.linspace(
        float(env.action_space.low[0]), float(env.action_space.high[0]),
        args.action_points, device=device
    ).unsqueeze(1)
    for index in np.flatnonzero((nominal_residual >= 0) | (qp_residual >= 0)):
        repeated = states[index:index + 1].expand(args.action_points, -1)
        scanned = corrected_residual(
            verifier, barrier, repeated, action_grid, args.batch_size
        )
        best = int(np.argmin(scanned))
        cases.append({
            "index_in_corrected_region": int(index),
            "state": states_np[index].tolist(),
            "barrier_value": float(values[mask][index].item()),
            "nominal_action": float(nominal_np[index]),
            "qp_action": float(qp_np[index]),
            "qp_slack": float(slack_np[index]),
            "nominal_corrected_residual": float(nominal_residual[index]),
            "qp_corrected_residual": float(qp_residual[index]),
            "qp_midpoint_residual": float(midpoint_residual[index]),
            "action_scan_nonpositive_count": int(np.sum(scanned <= 0)),
            "action_scan_best_action": float(action_grid[best, 0].item()),
            "action_scan_min_corrected_residual": float(scanned[best]),
        })

    metrics = {
        "experiment": "semantic_spvc_corrected_violation_diagnostic",
        "level": "corrected_region_and_finite_action_scan_not_a_certificate",
        "original_grid_states": total,
        "corrected_region_states": int(len(states)),
        "expected_region_states_from_recheck": int(recheck["original_filtered_region_states"]),
        "nominal_nonnegative_residuals": int(np.sum(nominal_residual >= 0)),
        "qp_nonnegative_residuals": int(np.sum(qp_residual >= 0)),
        "positive_slack_states": int(np.sum(slack_np > 1e-6)),
        "action_scan_points": args.action_points,
        "cases": cases,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(
        args.output_dir / "samples.npz", states=states_np,
        nominal_actions=nominal_np, qp_actions=qp_np, qp_slack=slack_np,
        nominal_corrected_residual=nominal_residual,
        qp_corrected_residual=qp_residual,
    )

    print("[Semantic-SPVC corrected violation diagnosis: no training]")
    print("region=%d (previous recheck=%d), nominal violations=%d, QP violations=%d" % (
        len(states), metrics["expected_region_states_from_recheck"],
        metrics["nominal_nonnegative_residuals"], metrics["qp_nonnegative_residuals"]
    ))
    print("positive QP slack=%d, detailed cases=%d" % (
        metrics["positive_slack_states"], len(cases)
    ))
    for case in cases:
        print("state=%s nominal=%.6f QP=%.6f slack=%.6f "
              "corrected residual %.6f -> %.6f, scan minimum=%.6f at action=%.3f" % (
                  case["state"], case["nominal_action"], case["qp_action"],
                  case["qp_slack"], case["nominal_corrected_residual"],
                  case["qp_corrected_residual"],
                  case["action_scan_min_corrected_residual"],
                  case["action_scan_best_action"],
              ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
