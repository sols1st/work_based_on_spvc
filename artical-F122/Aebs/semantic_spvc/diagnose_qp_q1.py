"""Explain Q1 counterexamples without training or changing the saved policy.

This is a sampled action-grid diagnostic, not a certificate. In particular,
"no feasible action on this grid" does not prove continuous-action infeasibility.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.semantic_spvc.check_qp import original_ibp_upper
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP
from Aebs.VT.verify import VTVerifier


def load_original_region(checkpoint_dir: Path):
    with open(checkpoint_dir / "metrics.json", encoding="utf-8") as input_file:
        baseline = json.load(input_file)
    info = baseline.get("loop_info", {})
    try:
        domain_min = float(info["domain_min"])
        unsafe_min = float(info["lb_unsafe"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Baseline metrics.json needs loop_info.domain_min and "
            "loop_info.lb_unsafe from the same saved SBC checkpoint"
        ) from error
    if not np.isfinite(domain_min) or not np.isfinite(unsafe_min) or unsafe_min <= domain_min:
        raise ValueError("Invalid original verifier region bounds")
    # Algebraically identical to VTVerifier.check_dec_cond()'s normalized test.
    return domain_min, unsafe_min, domain_min + 0.95 * (unsafe_min - domain_min)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1-dir", type=Path, default=Path("results/semantic_spvc_qp_q1"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/semantic_spvc_qp_q1b"))
    parser.add_argument("--action-points", type=int, default=121)
    parser.add_argument("--ibp-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.action_points < 3 or args.ibp_batch_size < 1:
        parser.error("--action-points must be >= 3 and --ibp-batch-size must be >= 1")
    if args.output_dir.resolve() in (args.q1_dir.resolve(), args.checkpoint_dir.resolve()):
        parser.error("--output-dir must differ from the input directories")

    with np.load(args.q1_dir / "samples.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    required = (
        "states", "nominal_actions", "qp_actions", "slack",
        "qp_midpoint_residual", "qp_ibp_residual",
    )
    for name in required:
        if name not in arrays:
            raise ValueError("Missing Q1 array: " + name)
    n = len(arrays["states"])
    if any(len(arrays[name]) != n for name in required) or arrays["states"].shape != (n, 2):
        raise ValueError("Q1 arrays have inconsistent shapes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    barrier.eval()
    for parameter in barrier.parameters():
        parameter.requires_grad_(False)
    domain_min, unsafe_min, region_low = load_original_region(args.checkpoint_dir)

    states = torch.as_tensor(arrays["states"], dtype=torch.float32, device=device)
    with torch.no_grad():
        barrier_values = barrier(states).flatten().cpu().numpy()
    in_original_region = (barrier_values > region_low) & (barrier_values < unsafe_min)

    nominal = arrays["nominal_actions"].reshape(-1)
    qp_action = arrays["qp_actions"].reshape(-1)
    low = float(env.action_space.low[0])
    high = float(env.action_space.high[0])
    clipped_nominal = np.clip(nominal, low, high)
    tolerance = 1e-6
    changed = np.abs(qp_action - nominal) > tolerance
    changed_beyond_clipping = np.abs(qp_action - clipped_nominal) > tolerance
    failing = (
        (arrays["qp_midpoint_residual"].reshape(-1) > tolerance)
        | (arrays["qp_ibp_residual"].reshape(-1) > tolerance)
        | (arrays["slack"].reshape(-1) > tolerance)
    )

    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = VTVerifier(
        SimpleNamespace(l_model=barrier), env, bounded,
        batch_size=args.ibp_batch_size, reach_prob=0.9, fail_check_fast=False,
    )
    action_grid = torch.linspace(low, high, args.action_points, device=device).unsqueeze(1)
    cases = []
    for index in np.flatnonzero(failing):
        repeated_state = states[index:index + 1].expand(args.action_points, -1)
        with torch.no_grad():
            midpoint = constraint.residual(repeated_state, action_grid).flatten().cpu().numpy()
            current = barrier(repeated_state)
            ibp = (original_ibp_upper(verifier, repeated_state, action_grid,
                                      args.ibp_batch_size) - current).flatten().cpu().numpy()
        midpoint_best = int(np.argmin(midpoint))
        ibp_best = int(np.argmin(ibp))
        case = {
            "index": int(index),
            "state": arrays["states"][index].tolist(),
            "barrier_value": float(barrier_values[index]),
            "in_original_verifier_region": bool(in_original_region[index]),
            "nominal_action": float(nominal[index]),
            "qp_action": float(qp_action[index]),
            "slack": float(arrays["slack"].reshape(-1)[index]),
            "qp_midpoint_residual": float(arrays["qp_midpoint_residual"].reshape(-1)[index]),
            "qp_ibp_residual": float(arrays["qp_ibp_residual"].reshape(-1)[index]),
            "action_grid_midpoint": {
                "nonpositive_count": int(np.sum(midpoint <= 0)),
                "minimum_residual": float(midpoint[midpoint_best]),
                "best_action": float(action_grid[midpoint_best, 0].item()),
            },
            "action_grid_ibp": {
                "nonpositive_count": int(np.sum(ibp <= 0)),
                "minimum_residual": float(ibp[ibp_best]),
                "best_action": float(action_grid[ibp_best, 0].item()),
            },
        }
        cases.append(case)

    metrics = {
        "experiment": "semantic_spvc_qp_q1b_failure_diagnostic",
        "level": "sampled_action_grid_diagnostic_not_a_certificate",
        "samples": int(n),
        "original_region": {
            "domain_min": domain_min,
            "unsafe_min": unsafe_min,
            "barrier_lower_exclusive": region_low,
            "state_count": int(np.sum(in_original_region)),
        },
        "actions": {
            "changed_from_raw_nominal": int(np.sum(changed)),
            "raw_nominal_out_of_bounds": int(np.sum(np.abs(nominal - clipped_nominal) > tolerance)),
            "changed_beyond_action_clipping": int(np.sum(changed_beyond_clipping)),
        },
        "failing_cases": cases,
        "action_grid_points": args.action_points,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)

    print("[Semantic-SPVC QP Q1b: failure diagnosis, not a certificate]")
    print("changed=%d/%d, nominal out of bounds=%d, beyond clipping=%d" % (
        metrics["actions"]["changed_from_raw_nominal"], n,
        metrics["actions"]["raw_nominal_out_of_bounds"],
        metrics["actions"]["changed_beyond_action_clipping"],
    ))
    print("in original verifier region=%d/%d, failing cases=%d" % (
        int(np.sum(in_original_region)), n, len(cases)
    ))
    for case in cases:
        print("index=%d state=%s region=%s slack=%.6f midpoint_min=%.6f ibp_min=%.6f" % (
            case["index"], case["state"], case["in_original_verifier_region"],
            case["slack"], case["action_grid_midpoint"]["minimum_residual"],
            case["action_grid_ibp"]["minimum_residual"],
        ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
