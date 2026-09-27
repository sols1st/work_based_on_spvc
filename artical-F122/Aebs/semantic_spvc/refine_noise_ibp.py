"""Check whether finer noise-cell IBP resolves a saved one-state violation.

Read-only sensitivity diagnostic. Dynamics, disturbance support, SBC weights,
and actions stay fixed; only the interval partition resolution changes.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from auto_LiRPA import BoundedModule

from Aebs.semantic_spvc.corrected_verify import CorrectedIntervalVTVerifier
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", type=Path,
                        default=Path("results/semantic_spvc_corrected_violation_diagnostic"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_noise_ibp_refinement"))
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--noise-bins", default="10,20,40,80",
                        help="Comma-separated number of cells per noise dimension")
    args = parser.parse_args()
    if args.output_dir.resolve() in (args.diagnostic_dir.resolve(), args.checkpoint_dir.resolve()):
        parser.error("--output-dir must differ from input directories")
    try:
        bins_list = [int(item.strip()) for item in args.noise_bins.split(",")]
    except ValueError as error:
        parser.error("--noise-bins must be comma-separated positive integers: %s" % error)
    if not bins_list or any(value < 1 or value > 100 for value in bins_list):
        parser.error("Each --noise-bins value must be in [1, 100]")

    with open(args.diagnostic_dir / "metrics.json", encoding="utf-8") as input_file:
        previous = json.load(input_file)
    cases = previous.get("cases", [])
    if not 0 <= args.case_index < len(cases):
        parser.error("--case-index must select a case in the previous diagnostic")
    case = cases[args.case_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    barrier.eval()
    for parameter in barrier.parameters():
        parameter.requires_grad_(False)
    bounded = BoundedModule(barrier, torch.zeros((1, 2), device=device), device=device)
    verifier = CorrectedIntervalVTVerifier(
        SimpleNamespace(l_model=barrier), env, bounded,
        batch_size=1, reach_prob=0.9, fail_check_fast=False,
    )
    state = torch.tensor([case["state"]], dtype=torch.float32, device=device)
    actions = {
        "nominal": float(case["nominal_action"]),
        "qp": float(case["qp_action"]),
        "best_from_121_action_scan": float(case["action_scan_best_action"]),
    }
    with torch.no_grad():
        current = float(barrier(state).item())
    rows = []
    for bins in bins_list:
        verifier.pmass_n = np.full(2, bins, dtype=int)
        verifier._cached_pmass_grid = None
        pmass, noise_low, noise_high = verifier.get_pmass_grid()
        midpoint = DiscreteSBCConstraint(barrier, env, noise_bins=bins).to(device)
        row = {
            "noise_bins_per_dimension": bins,
            "noise_cells": bins * bins,
            "probability_mass_sum": float(pmass.sum().item()),
            "actions": {},
        }
        for label, number in actions.items():
            action = torch.tensor([[number]], dtype=torch.float32, device=device)
            with torch.no_grad():
                upper = verifier.compute_expected_l(
                    state, action, pmass, noise_low, noise_high
                )
                midpoint_residual = midpoint.residual(state, action)
            row["actions"][label] = {
                "action": number,
                "corrected_ibp_residual": float(upper.item() - current),
                "midpoint_residual": float(midpoint_residual.item()),
            }
        rows.append(row)

    metrics = {
        "experiment": "semantic_spvc_noise_ibp_refinement",
        "level": "single_state_partition_sensitivity_not_a_certificate",
        "state": case["state"],
        "barrier_value": current,
        "actions_unchanged": actions,
        "same_noise_support": True,
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)

    print("[Semantic-SPVC corrected IBP noise-grid refinement: one-state diagnostic]")
    print("state=%s, barrier=%.6f" % (case["state"], current))
    for row in rows:
        nominal = row["actions"]["nominal"]
        best = row["actions"]["best_from_121_action_scan"]
        print("%dx%d cells: nominal IBP=%.6f, midpoint=%.6f; "
              "best-scan-action IBP=%.6f, midpoint=%.6f" % (
                  row["noise_bins_per_dimension"], row["noise_bins_per_dimension"],
                  nominal["corrected_ibp_residual"], nominal["midpoint_residual"],
                  best["corrected_ibp_residual"], best["midpoint_residual"],
              ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
