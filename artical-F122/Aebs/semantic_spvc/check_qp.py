"""Q1 diagnostic for a frozen semantic PPO/SBC and a differentiable scalar QP.

This command does not train or run episodes. It samples the original VT state
grid, compares nominal and QP actions, and rechecks the nonlinear midpoint
expectation and the original verifier's reported IBP value. Q1c exposed that
the latter may not be an interval upper bound; use audit_ibp.py to check it.
This command does not certify the state domain or image-to-semantics path.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP
from Aebs.VT.verify import VTVerifier


def grid_sample(env: Aebs, count: int) -> np.ndarray:
    size = env.space_split
    total = int(np.prod(size))
    if count < 1 or count > total:
        raise ValueError("samples must be between 1 and the original grid size")
    axes = [
        np.linspace(low, high, int(n), endpoint=False, dtype=np.float32)
        + (high - low) / (2 * int(n))
        for low, high, n in zip(env.observation_space.low, env.observation_space.high, size)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    complete = np.stack(mesh, axis=-1).reshape(-1, 2)
    return complete[np.linspace(0, total - 1, count, dtype=np.int64)]


def original_verifier_region_sample(env: Aebs, barrier: torch.nn.Module,
                                    checkpoint_dir: Path, count: int,
                                    device: torch.device):
    """Select grid centers using the exact state mask in VT.check_dec_cond()."""
    with open(checkpoint_dir / "metrics.json", encoding="utf-8") as input_file:
        baseline = json.load(input_file)
    info = baseline.get("loop_info", {})
    try:
        domain_min = float(info["domain_min"])
        unsafe_min = float(info["lb_unsafe"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Baseline metrics.json needs loop_info.domain_min and "
            "loop_info.lb_unsafe from this SBC checkpoint"
        ) from error
    if not np.isfinite(domain_min) or not np.isfinite(unsafe_min) or unsafe_min <= domain_min:
        raise ValueError("Invalid original verifier region bounds")

    total = int(np.prod(env.space_split))
    grid = grid_sample(env, total)
    values = []
    with torch.no_grad():
        for start in range(0, total, 2048):
            batch = torch.as_tensor(grid[start:start + 2048], device=device)
            values.append(barrier(batch).flatten().cpu().numpy())
    barrier_values = np.concatenate(values)
    # Equivalent to normalized_l > 0.95*normalized_unsafe_min and
    # normalized_l < normalized_unsafe_min in the original VT verifier.
    region_low = domain_min + 0.95 * (unsafe_min - domain_min)
    eligible = np.flatnonzero((barrier_values > region_low) & (barrier_values < unsafe_min))
    if len(eligible) == 0:
        raise RuntimeError("Original verifier region contains no grid centers")
    selected = eligible[np.linspace(0, len(eligible) - 1,
                                    min(count, len(eligible)), dtype=np.int64)]
    return grid[selected], int(len(eligible))


def summary(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def original_ibp_upper(verifier: VTVerifier, states: torch.Tensor,
                       actions: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reproduce the original VT result, without claiming it is a valid upper bound."""
    pmass, noise_low, noise_high = verifier.get_pmass_grid()
    parts = []
    with torch.no_grad():
        for start in range(0, states.shape[0], batch_size):
            end = min(start + batch_size, states.shape[0])
            upper = verifier.compute_expected_l(
                states[start:end], actions[start:end], pmass, noise_low, noise_high
            )
            parts.append(upper.reshape(-1, 1))
    return torch.cat(parts, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_q1"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--scope", choices=("uniform", "original-verifier"),
                        default="uniform", help="Which original state-grid centers to diagnose")
    parser.add_argument("--noise-bins", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.0,
                        help="0 matches the original hard decrease count; positive values are diagnostic margins")
    parser.add_argument("--slack-weight", type=float, default=100.0)
    parser.add_argument("--ibp-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.ibp_batch_size < 1:
        parser.error("--ibp-batch-size must be positive")
    if args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("--output-dir must differ from --checkpoint-dir")

    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    if not 1 <= args.samples <= int(np.prod(env.space_split)):
        parser.error("--samples must be between 1 and the original grid size")

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

    full_region_count = None
    if args.scope == "original-verifier":
        selected_states, full_region_count = original_verifier_region_sample(
            env, barrier, args.checkpoint_dir, args.samples, device
        )
    else:
        selected_states = grid_sample(env, args.samples)
    states = torch.as_tensor(selected_states, device=device)

    with torch.no_grad():
        latent = torch.zeros((states.shape[0], 4), dtype=states.dtype, device=device)
        nominal = policy(latent, states)

    constraint = DiscreteSBCConstraint(
        barrier, env, noise_bins=args.noise_bins, margin=args.margin
    ).to(device)
    qp = ScalarSBCQP(
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]),
        slack_weight=args.slack_weight,
    ).to(device)
    affine = constraint.linearize(states, nominal)
    solution = qp(nominal, affine.coefficient, affine.right_hand_side)

    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = VTVerifier(
        SimpleNamespace(l_model=barrier), env, bounded,
        batch_size=args.ibp_batch_size, reach_prob=0.9, fail_check_fast=False,
    )
    with torch.no_grad():
        nominal_midpoint = constraint.residual(states, nominal)
        safe_midpoint = constraint.residual(states, solution.action)
        current_value = barrier(states)
        nominal_ibp = original_ibp_upper(
            verifier, states, nominal, args.ibp_batch_size
        ) - current_value + args.margin
        safe_ibp = original_ibp_upper(
            verifier, states, solution.action, args.ibp_batch_size
        ) - current_value + args.margin

    # Synthetic active-constraint probe: tests that this QP implementation
    # actually propagates a nonzero gradient through its action output.
    probe_nominal = torch.tensor([[1.0]], device=device, requires_grad=True)
    probe_action = qp(
        probe_nominal, torch.tensor([[1.0]], device=device),
        torch.tensor([[0.0]], device=device)
    ).action
    probe_gradient = torch.autograd.grad(probe_action.sum(), probe_nominal)[0]

    arrays = {
        "states": states.detach().cpu().numpy(),
        "nominal_actions": nominal.detach().cpu().numpy(),
        "qp_actions": solution.action.detach().cpu().numpy(),
        "slack": solution.slack.detach().cpu().numpy(),
        "affine_coefficient": affine.coefficient.detach().cpu().numpy(),
        "affine_rhs": affine.right_hand_side.detach().cpu().numpy(),
        "nominal_midpoint_residual": nominal_midpoint.cpu().numpy(),
        "qp_midpoint_residual": safe_midpoint.cpu().numpy(),
        "nominal_ibp_residual": nominal_ibp.cpu().numpy(),
        "qp_ibp_residual": safe_ibp.cpu().numpy(),
    }
    arrays["qp_affine_residual_without_slack"] = (
        affine.coefficient * solution.action - affine.right_hand_side
    ).detach().cpu().numpy()
    arrays["linearization_error"] = (
        arrays["qp_midpoint_residual"] - arrays["qp_affine_residual_without_slack"]
    )
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("QP diagnostic produced nonfinite values")

    action_change = np.abs(arrays["qp_actions"] - arrays["nominal_actions"])
    slack = arrays["slack"]
    threshold = 1e-6
    metrics = {
        "experiment": "semantic_spvc_qp_q1",
        "level": "sampled_grid_diagnostic_not_a_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "samples": int(states.shape[0]),
        "scope": args.scope,
        "full_original_verifier_region_count": full_region_count,
        "noise_midpoint_bins_per_dimension": args.noise_bins,
        "ibp_noise_bins_per_dimension": 10,
        "margin": args.margin,
        "slack_weight": args.slack_weight,
        "device": str(device),
        "nominal_action": summary(arrays["nominal_actions"]),
        "qp_action": summary(arrays["qp_actions"]),
        "absolute_action_change": summary(action_change),
        "intervention_rate": float(np.mean(action_change > threshold)),
        "slack": summary(slack),
        "positive_slack_rate": float(np.mean(slack > threshold)),
        "qp_solver_failures": 0,
        "flat_slope_rate": float(np.mean(np.abs(arrays["affine_coefficient"]) <= 1e-9)),
        "affine_linear_residual_max": float(solution.linear_residual.detach().max().cpu()),
        "linearization_error": summary(arrays["linearization_error"]),
        "midpoint": {
            "nominal_positive_residual_count": int(np.sum(arrays["nominal_midpoint_residual"] > 0)),
            "qp_positive_residual_count": int(np.sum(arrays["qp_midpoint_residual"] > 0)),
            "qp_residual": summary(arrays["qp_midpoint_residual"]),
        },
        "original_verifier_ibp_upper": {
            "nominal_positive_residual_count": int(np.sum(arrays["nominal_ibp_residual"] > 0)),
            "qp_positive_residual_count": int(np.sum(arrays["qp_ibp_residual"] > 0)),
            "qp_residual": summary(arrays["qp_ibp_residual"]),
        },
        "gradient_probe": {
            "nominal_to_qp_action": float(probe_gradient.item()),
            "finite": bool(torch.isfinite(probe_gradient).all().item()),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "samples.npz", **arrays)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2, ensure_ascii=False, allow_nan=False)

    print("[Semantic-SPVC QP Q1: sampled diagnostic, not a certificate]")
    print("scope=%s, full original verifier region states=%s" % (
        args.scope, full_region_count if full_region_count is not None else "not counted"
    ))
    print("states=%d, midpoint noise cells=%d, IBP noise cells=100" % (
        states.shape[0], args.noise_bins ** 2
    ))
    print("QP intervention=%.2f%%, positive slack=%.2f%%, max slack=%.6f" % (
        100 * metrics["intervention_rate"], 100 * metrics["positive_slack_rate"],
        metrics["slack"]["max"]
    ))
    print("midpoint positive residual: nominal=%d, QP=%d" % (
        metrics["midpoint"]["nominal_positive_residual_count"],
        metrics["midpoint"]["qp_positive_residual_count"]
    ))
    print("legacy verifier-reported IBP positive residual: nominal=%d, QP=%d" % (
        metrics["original_verifier_ibp_upper"]["nominal_positive_residual_count"],
        metrics["original_verifier_ibp_upper"]["qp_positive_residual_count"]
    ))
    print("linearization error: mean=%.6f, max=%.6f" % (
        metrics["linearization_error"]["mean"], metrics["linearization_error"]["max"]
    ))
    print("QP output gradient wrt nominal=%.6f" % metrics["gradient_probe"]["nominal_to_qp_action"])
    print("metrics: %s" % (args.output_dir / "metrics.json"))
    print("per-state data: %s" % (args.output_dir / "samples.npz"))


if __name__ == "__main__":
    main()
