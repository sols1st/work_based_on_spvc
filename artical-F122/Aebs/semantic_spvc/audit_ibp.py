"""Audit the interval meaning of the original VT bound call on Q1c states.

This does not train, modify checkpoints, or certify the complete state grid.
It compares the legacy two-tensor call with a proper BoundedTensor interval.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

from Aebs.semantic_spvc.check_qp import original_ibp_upper
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP
from Aebs.VT.verify import VTVerifier


def interval_expectation(verifier, bounded, env, states, actions, batch_size):
    """Weighted IBP upper bound using one interval-perturbed model input."""
    pmass, noise_low, noise_high = verifier.get_pmass_grid()
    count = noise_low.shape[0]
    pieces = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            s = states[start:start + batch_size]
            a = actions[start:start + batch_size]
            next_state = env.v_next(s, a)
            lower = (next_state[:, None, :] + noise_low[None, :, :]).reshape(-1, 2)
            upper = (next_state[:, None, :] + noise_high[None, :, :]).reshape(-1, 2)
            center = (lower + upper) / 2
            perturbation = PerturbationLpNorm(norm=np.inf, x_L=lower, x_U=upper)
            bounded_input = BoundedTensor(center, perturbation)
            _, cell_upper = bounded.compute_bounds(x=(bounded_input,), method="IBP")
            cell_upper = cell_upper.reshape(len(s), count)
            correct_expected = (cell_upper * pmass.reshape(1, count)).sum(dim=1, keepdim=True)
            pieces.append(correct_expected)
    return torch.cat(pieces, dim=0)


def lower_corner_expectation(verifier, barrier, env, states, actions, batch_size):
    """Weighted plain model value at each noise-cell lower corner."""
    pmass, noise_low, _ = verifier.get_pmass_grid()
    count = noise_low.shape[0]
    pieces = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            s = states[start:start + batch_size]
            a = actions[start:start + batch_size]
            lower = (env.v_next(s, a)[:, None, :] + noise_low[None, :, :]).reshape(-1, 2)
            values = barrier(lower).reshape(len(s), count)
            pieces.append((values * pmass.reshape(1, count)).sum(dim=1, keepdim=True))
    return torch.cat(pieces, dim=0)


def summarize(values):
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q1c-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_q1c_region"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_q1d_ibp_audit"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output_dir.resolve() in (args.q1c_dir.resolve(), args.checkpoint_dir.resolve()):
        parser.error("--output-dir must differ from input directories")

    with np.load(args.q1c_dir / "samples.npz") as saved:
        states_np = saved["states"]
        nominal_np = saved["nominal_actions"]
        qp_np = saved["qp_actions"]
        old_qp_np = saved["qp_ibp_residual"]
        midpoint_qp_np = saved["qp_midpoint_residual"]
    if states_np.ndim != 2 or states_np.shape[1] != 2 or not len(states_np):
        raise ValueError("Q1c state array is empty or malformed")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    barrier.eval()
    for parameter in barrier.parameters():
        parameter.requires_grad_(False)
    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = VTVerifier(
        SimpleNamespace(l_model=barrier), env, bounded,
        batch_size=args.batch_size, reach_prob=0.9, fail_check_fast=False,
    )
    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    current = barrier(states).detach()

    results = {}
    for name, actions_np in (("nominal", nominal_np), ("qp", qp_np)):
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
        old = original_ibp_upper(verifier, states, actions, args.batch_size) - current
        corner = lower_corner_expectation(
            verifier, barrier, env, states, actions, args.batch_size
        ) - current
        correct = interval_expectation(
            verifier, bounded, env, states, actions, args.batch_size
        ) - current
        midpoint = constraint.residual(states, actions).detach()
        arrays = {
            "legacy_residual": old.detach().cpu().numpy().reshape(-1),
            "lower_corner_residual": corner.detach().cpu().numpy().reshape(-1),
            "correct_interval_residual": correct.detach().cpu().numpy().reshape(-1),
            "midpoint_residual": midpoint.detach().cpu().numpy().reshape(-1),
        }
        legacy_corner_gap = arrays["legacy_residual"] - arrays["lower_corner_residual"]
        interval_midpoint_gap = arrays["correct_interval_residual"] - arrays["midpoint_residual"]
        results[name] = {
            "legacy_positive": int(np.sum(arrays["legacy_residual"] > 0)),
            "correct_interval_positive": int(np.sum(arrays["correct_interval_residual"] > 0)),
            "midpoint_positive": int(np.sum(arrays["midpoint_residual"] > 0)),
            "legacy_minus_lower_corner_max_abs": float(np.max(np.abs(legacy_corner_gap))),
            "correct_interval_minus_midpoint": summarize(interval_midpoint_gap),
            "legacy_residual": summarize(arrays["legacy_residual"]),
            "correct_interval_residual": summarize(arrays["correct_interval_residual"]),
            "arrays": arrays,
        }

    qp = results["qp"]
    saved_gap = np.max(np.abs(qp["arrays"]["legacy_residual"] - old_qp_np.reshape(-1)))
    midpoint_gap = np.max(np.abs(qp["arrays"]["midpoint_residual"] - midpoint_qp_np.reshape(-1)))
    if saved_gap > 1e-3 or midpoint_gap > 1e-3:
        raise RuntimeError("Audit does not match Q1c arrays; check that checkpoint files are unchanged")

    metrics = {
        "experiment": "semantic_spvc_qp_q1d_ibp_audit",
        "level": "sampled_interval_implementation_audit_not_a_full_certificate",
        "states": int(len(states)),
        "noise_cells": 100,
        "saved_q1c_legacy_max_abs_difference": float(saved_gap),
        "saved_q1c_midpoint_max_abs_difference": float(midpoint_gap),
    }
    for name, result in results.items():
        metrics[name] = {key: value for key, value in result.items() if key != "arrays"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    np.savez_compressed(args.output_dir / "samples.npz", states=states_np,
                        **{name + "_" + key: value
                           for name, result in results.items()
                           for key, value in result["arrays"].items()})

    print("[Semantic-SPVC QP Q1d: interval implementation audit, not a full certificate]")
    print("states=%d, noise cells=100" % len(states))
    for name in ("nominal", "qp"):
        item = metrics[name]
        print("%s: legacy positive=%d, corrected interval positive=%d, midpoint positive=%d" % (
            name, item["legacy_positive"], item["correct_interval_positive"],
            item["midpoint_positive"]
        ))
        print("%s: legacy/lower-corner max difference=%.8f, corrected-minus-midpoint min=%.8f" % (
            name, item["legacy_minus_lower_corner_max_abs"],
            item["correct_interval_minus_midpoint"]["min"]
        ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
