"""Minimal paired rollout: saved semantic-SPVC PPO versus frozen PPO+SBC-QP.

Uses the original AEBS environment and saved weights without training. The QP
constraint is the existing differentiable midpoint linearization. This is an
empirical comparison, not an SPVC safety verification or image-path test.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Aebs.semantic_spvc.model import SemanticSPVCPolicy
from Aebs.semantic_spvc.qp_constraint import DiscreteSBCConstraint
from Aebs.semantic_spvc.qp_layer import ScalarSBCQP
from Aebs.system.env import Aebs, AebsEnv
from Aebs.system.outcomes import EPISODE_OUTCOMES, TIMEOUT
from Aebs.VT.utils import MLP


def initial_states(distance_points, speed_points):
    distances = 15.0 + (np.arange(distance_points) + 0.5) / distance_points
    speeds = 2.5 + 0.5 * (np.arange(speed_points) + 0.5) / speed_points
    return [(float(d), float(v)) for d in distances for v in speeds]


def summarize(episodes):
    count = len(episodes)
    outcomes = {name: sum(row["outcome"] == name for row in episodes)
                for name in EPISODE_OUTCOMES}
    return {
        "episodes": count,
        "outcomes": outcomes,
        "success_rate": outcomes["success"] / count,
        "unsafe_rate": outcomes["unsafe"] / count,
        "timeout_rate": outcomes[TIMEOUT] / count,
        "mean_return": float(np.mean([row["return"] for row in episodes])),
        "mean_steps": float(np.mean([row["steps"] for row in episodes])),
        "raw_action_out_of_bounds_rate": float(
            sum(row["raw_action_out_of_bounds"] for row in episodes)
            / max(1, sum(row["steps"] for row in episodes))
        ),
        "qp_intervention_rate": float(
            sum(row["qp_interventions"] for row in episodes)
            / max(1, sum(row["steps"] for row in episodes))
        ),
        "qp_change_beyond_clipping_rate": float(
            sum(row["qp_changes_beyond_clipping"] for row in episodes)
            / max(1, sum(row["steps"] for row in episodes))
        ),
        "positive_slack_rate": float(
            sum(row["positive_slack_steps"] for row in episodes)
            / max(1, sum(row["steps"] for row in episodes))
        ),
        "maximum_slack": float(max(row["maximum_slack"] for row in episodes)),
    }


def rollout(env, policy, barrier_constraint, qp, state_m, mode, device):
    distance_m, speed = state_m
    simulator = AebsEnv(env.std1, max_episode_steps=400)
    simulator.reset()
    simulator.state = np.array([distance_m / env.std1, speed], dtype=np.float32)
    simulator.elapsed_steps = 0
    result = {
        "initial_distance_m": distance_m,
        "initial_speed_mps": speed,
        "return": 0.0,
        "steps": 0,
        "outcome": TIMEOUT,
        "raw_action_out_of_bounds": 0,
        "qp_interventions": 0,
        "qp_changes_beyond_clipping": 0,
        "positive_slack_steps": 0,
        "maximum_slack": 0.0,
    }
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])
    for _ in range(simulator.max_episode_steps):
        state = torch.as_tensor(simulator.state[None, :], dtype=torch.float32,
                                device=device)
        latent = torch.zeros((1, 4), dtype=torch.float32, device=device)
        with torch.no_grad():
            nominal = policy(latent, state)
        raw_action = float(nominal.item())
        clipped_action = float(np.clip(raw_action, action_low, action_high))
        if abs(raw_action - clipped_action) > 1e-6:
            result["raw_action_out_of_bounds"] += 1

        if mode == "qp":
            affine = barrier_constraint.linearize(state, nominal)
            solution = qp(nominal, affine.coefficient, affine.right_hand_side)
            actual_action = float(solution.action.detach().item())
            slack = float(solution.slack.detach().item())
            if abs(actual_action - raw_action) > 1e-6:
                result["qp_interventions"] += 1
            if abs(actual_action - clipped_action) > 1e-6:
                result["qp_changes_beyond_clipping"] += 1
            if slack > 1e-6:
                result["positive_slack_steps"] += 1
            result["maximum_slack"] = max(result["maximum_slack"], slack)
        else:
            actual_action = clipped_action

        _, reward, terminated, truncated, info = simulator.step(
            np.array([actual_action], dtype=np.float32)
        )
        result["return"] += float(reward)
        result["steps"] += 1
        if terminated or truncated:
            result["outcome"] = info.get("outcome", TIMEOUT)
            break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/semantic_spvc_my_run"))
    parser.add_argument("--semantic-checkpoint",
                        default="results/mvp/01_semantic/semantic_encoder.pt")
    parser.add_argument("--controller", default="Aebs/controller/best_model/best_model.zip")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/semantic_spvc_qp_q2_original"))
    parser.add_argument("--distance-points", type=int, default=5)
    parser.add_argument("--speed-points", type=int, default=4)
    parser.add_argument("--slack-weight", type=float, default=100.0)
    parser.add_argument("--use-original-controller", action="store_true",
                        help="Keep the original pretrained PPO weights; load only SBC from checkpoint-dir")
    args = parser.parse_args()
    if args.distance_points < 1 or args.speed_points < 1:
        parser.error("--distance-points and --speed-points must be positive")
    if args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("--output-dir must differ from --checkpoint-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Aebs(0.05)
    policy = SemanticSPVCPolicy.from_checkpoints(
        args.semantic_checkpoint, args.controller, device
    )
    if not args.use_original_controller:
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
    constraint = DiscreteSBCConstraint(barrier, env, noise_bins=10).to(device)
    qp = ScalarSBCQP(
        action_low=float(env.action_space.low[0]),
        action_high=float(env.action_space.high[0]),
        slack_weight=args.slack_weight,
    ).to(device)

    starts = initial_states(args.distance_points, args.speed_points)
    baseline = []
    filtered = []
    for index, start in enumerate(starts, 1):
        baseline.append(rollout(env, policy, constraint, qp, start, "baseline", device))
        filtered.append(rollout(env, policy, constraint, qp, start, "qp", device))
        print("paired episode %d/%d: PPO=%s, QP=%s" % (
            index, len(starts), baseline[-1]["outcome"], filtered[-1]["outcome"]
        ), flush=True)

    metrics = {
        "experiment": "semantic_spvc_qp_q2_original",
        "level": "paired_rollout_empirical_only_not_a_certificate",
        "checkpoint_dir": str(args.checkpoint_dir),
        "controller_source": ("original_pretrained_ppo" if args.use_original_controller
                              else str(args.checkpoint_dir / "semantic_ppo.pt")),
        "state_path": "true semantic state -> saved PPO -> optional frozen SBC-QP -> original AEBS environment",
        "image_path_tested": False,
        "initial_region_distance_m": [15.0, 16.0],
        "initial_region_speed_mps": [2.5, 3.0],
        "paired_initial_states": len(starts),
        "qp_constraint": "10x10 midpoint linearization, soft slack, no training",
        "slack_weight": args.slack_weight,
        "baseline": summarize(baseline),
        "qp": summarize(filtered),
        "episodes": {"baseline": baseline, "qp": filtered},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=2, ensure_ascii=False, allow_nan=False)

    print("[Semantic-SPVC QP Q2: original AEBS paired rollout, no training]")
    for name in ("baseline", "qp"):
        item = metrics[name]
        print("%s: success=%.2f%%, unsafe=%.2f%%, timeout=%.2f%%, "
              "return=%.3f, steps=%.1f" % (
                  name, item["success_rate"] * 100, item["unsafe_rate"] * 100,
                  item["timeout_rate"] * 100, item["mean_return"],
                  item["mean_steps"],
              ))
    print("QP: intervention=%.2f%%, beyond clipping=%.2f%%, "
          "positive slack=%.2f%%, max slack=%.6f" % (
              metrics["qp"]["qp_intervention_rate"] * 100,
              metrics["qp"]["qp_change_beyond_clipping_rate"] * 100,
              metrics["qp"]["positive_slack_rate"] * 100,
              metrics["qp"]["maximum_slack"],
          ))
    print("metrics: %s" % (args.output_dir / "metrics.json"))


if __name__ == "__main__":
    main()
