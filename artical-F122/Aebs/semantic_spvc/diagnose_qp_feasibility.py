"""Diagnose SBC-QP feasibility and the original verifier's selected region.

For selected original-AEBS trajectories, scan legal actions against the same
10x10 midpoint SBC condition used by the QP. This is a diagnostic only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Aebs.semantic_spvc.evaluate_qp_original import initial_states
from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.system.env import Aebs, AebsEnv
from Aebs.VT.utils import MLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_joint_full_20260926"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_feasibility"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--action-points", type=int, default=61)
    args = parser.parse_args()
    if args.action_points < 3:
        parser.error("--action-points must be at least 3")
    if args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("--output-dir must differ from checkpoint directory")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint, args.controller, device
    )
    policy.load_state_dict(torch.load(args.checkpoint_dir / "semantic_ppo.pt", map_location=device))
    policy.eval()
    barrier = MLP([2, 16, 8, 1], activation="tanh", square_output=True).to(device)
    barrier.load_state_dict(torch.load(args.checkpoint_dir / "sbc.pt", map_location=device))
    barrier.eval()
    for model in (policy, barrier):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    qp = ScalarSBCQP(-3.0, 3.0, slack_weight=100.0).to(device)
    legacy = json.loads((args.checkpoint_dir / "metrics.json").read_text())[
        "legacy_spvc_verification"
    ]
    unsafe_lower = float(legacy["unsafe_lower"])
    domain_lower = float(legacy["domain_lower"])
    # Algebraically equivalent to VTVerifier's normalized 0.95..1.00 band
    # when its denominator is positive; these are legacy values, not sound bounds.
    region_low = 0.95 * unsafe_lower + 0.05 * domain_lower

    starts = initial_states(5, 4)
    selected_starts = [starts[index] for index in (0, 10, 19)]
    trajectories = []
    for distance_m, speed in selected_starts:
        sim = AebsEnv(env.std1, max_episode_steps=400)
        sim.reset()
        sim.state = np.array([distance_m / env.std1, speed], dtype=np.float32)
        sim.elapsed_steps = 0
        rows = []
        for _ in range(sim.max_episode_steps):
            state = torch.as_tensor(sim.state[None], dtype=torch.float32, device=device)
            z = torch.zeros((1, 4), dtype=torch.float32, device=device)
            with torch.no_grad():
                nominal = policy(z, state)
                barrier_value = float(barrier(state).item())
            affine = constraint.linearize(state, nominal)
            solution = qp(nominal, affine.coefficient, affine.right_hand_side)
            row = {
                "state": sim.state.tolist(),
                "nominal": float(nominal.item()),
                "qp_action": float(solution.action.detach().item()),
                "slack": float(solution.slack.detach().item()),
                "slope": float(affine.coefficient.detach().item()),
                "barrier_value": barrier_value,
                "in_original_verifier_region": region_low < barrier_value < unsafe_lower,
            }
            rows.append(row)
            _, _, terminated, truncated, info = sim.step(
                np.array([row["qp_action"]], dtype=np.float32)
            )
            if terminated or truncated:
                outcome = info.get("outcome", "timeout")
                break
        trajectories.append({"initial": [distance_m, speed], "outcome": outcome,
                             "steps": len(rows), "rows": rows})

    action_grid = torch.linspace(-3.0, 3.0, args.action_points, device=device).unsqueeze(1)
    examined = []
    for episode_index, trajectory in enumerate(trajectories):
        rows = trajectory["rows"]
        selected = {index for index, row in enumerate(rows) if row["slack"] > 1e-6}
        selected.update(range(max(0, len(rows) - 10), len(rows)))
        selected.update(range(0, len(rows), 5))
        for step in sorted(selected):
            row = rows[step]
            state = torch.tensor([row["state"]], dtype=torch.float32, device=device)
            repeated = state.expand(args.action_points, -1)
            with torch.no_grad():
                residuals = constraint.residual(repeated, action_grid).flatten().cpu().numpy()
            best = int(np.argmin(residuals))
            examined.append({
                "episode": episode_index, "step": step, "state": row["state"],
                "slack": row["slack"], "nominal": row["nominal"],
                "qp_action": row["qp_action"], "slope": row["slope"],
                "barrier_value": row["barrier_value"],
                "in_original_verifier_region": row["in_original_verifier_region"],
                "feasible_action_count": int(np.sum(residuals <= 0)),
                "minimum_residual": float(residuals[best]),
                "best_action": float(action_grid[best, 0].item()),
            })

    slack_cases = [row for row in examined if row["slack"] > 1e-6]
    metrics = {
        "experiment": "semantic_spvc_qp_feasibility",
        "level": "three_trajectory_finite_action_scan_not_a_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "trajectories": [{key: value for key, value in item.items() if key != "rows"}
                         for item in trajectories],
        "examined_states": len(examined),
        "slack_states": len(slack_cases),
        "slack_states_with_scanned_feasible_action": sum(
            item["feasible_action_count"] > 0 for item in slack_cases
        ),
        "slack_states_without_scanned_feasible_action": sum(
            item["feasible_action_count"] == 0 for item in slack_cases
        ),
        "legacy_verifier_region_barrier_range": [region_low, unsafe_lower],
        "all_trajectory_steps": sum(len(item["rows"]) for item in trajectories),
        "all_steps_in_original_verifier_region": sum(
            row["in_original_verifier_region"] for item in trajectories
            for row in item["rows"]
        ),
        "slack_states_in_original_verifier_region": sum(
            item["in_original_verifier_region"] for item in slack_cases
        ),
        "minimum_of_minimum_residuals": min(item["minimum_residual"] for item in examined),
        "maximum_of_minimum_residuals": max(item["minimum_residual"] for item in examined),
        "examples": examined[:10],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)
    print("[QP midpoint-constraint feasibility: three trajectories]")
    print("examined=%d, slack=%d, slack with scanned feasible action=%d, "
          "slack without scanned feasible action=%d" % (
              metrics["examined_states"], metrics["slack_states"],
              metrics["slack_states_with_scanned_feasible_action"],
              metrics["slack_states_without_scanned_feasible_action"],
          ))
    print("minimum residual among actions per state: range [%.6f, %.6f]" % (
        metrics["minimum_of_minimum_residuals"], metrics["maximum_of_minimum_residuals"]
    ))
    print("legacy verifier band steps=%d/%d, slack cases inside band=%d/%d" % (
        metrics["all_steps_in_original_verifier_region"],
        metrics["all_trajectory_steps"],
        metrics["slack_states_in_original_verifier_region"],
        metrics["slack_states"],
    ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
