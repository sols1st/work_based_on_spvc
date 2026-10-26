"""Frozen native PPO -> physical-state SBC interface and feasibility gate.

Uses the NEW normalized q, not R2's historical q or distance decoder.
Development radii are not misregistered as fresh CP contracts. A failing
allowed rollout blocks the strong pathwise-global-box barrier specification;
it does not show that every stochastic SBC formulation is impossible.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO

from Aebs.conformal.calibrate_controller import classify_batch
from Aebs.dino_latent.models import PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256


class FrozenNativeActor(nn.Module):
    """Physical [distance in meters, speed] -> own normalized q -> PPO mean."""
    def __init__(self, q, actor, distance_mean, distance_std):
        super().__init__()
        self.q, self.actor = q, actor
        self.register_buffer("distance_mean", torch.tensor(float(distance_mean)))
        self.register_buffer("distance_std", torch.tensor(float(distance_std)))
        if float(self.distance_std) <= 0:
            raise ValueError("Invalid distance standardization")
        self.requires_grad_(False); self.eval()

    def forward(self, states):
        z = self.q((states[:, :1] - self.distance_mean) / self.distance_std)
        return self.actor(torch.cat((z, states[:, 1:2]), -1)).clamp(-3, 3)


def load_frozen(native_dir, controller_dir):
    ck_path, controller_path = native_dir / "historical_R2.pt", controller_dir / "native_ppo.zip"
    record = json.loads((controller_dir / "metrics.json").read_text())
    if (record["name"] != "historical_R2" or record["status"] != "native_ppo_speed_residual_development_complete"
            or record["representation_sha256"] != sha256(ck_path)
            or record["controller_sha256"] != sha256(controller_path)):
        raise ValueError("Wrong frozen native PPO/q provenance")
    ck = torch.load(ck_path, map_location="cpu")
    q = PhysicalToLatent(latent_dim=int(ck["result"]["dimension"])).eval()
    q.load_state_dict(ck["q_state_dict"])
    ppo = PPO.load(controller_path, device="cpu")
    ppo.policy.set_training_mode(False)
    actor = nn.Sequential(copy.deepcopy(ppo.policy.mlp_extractor.policy_net), copy.deepcopy(ppo.policy.action_net))
    policy = FrozenNativeActor(q, actor, ck["distance_mean"], ck["distance_std"])
    states = np.random.default_rng(7).uniform([5, 0], [16, 3], size=(4096, 2)).astype(np.float32)
    with torch.no_grad():
        s = torch.from_numpy(states)
        z = q((s[:, :1] - float(ck["distance_mean"])) / float(ck["distance_std"]))
        expected = ppo.predict(torch.cat((z, s[:, 1:2]), 1).numpy(), deterministic=True)[0]
        actual = policy(s).numpy()
    maximum = float(np.max(abs(actual - expected)))
    if maximum > 1e-6:
        raise ValueError("Native SBC adapter differs from saved PPO")
    sources = {str(p): sha256(p) for p in (ck_path, controller_path, controller_dir / "metrics.json")}
    return policy, record, dict(source_sha256=sources, adapter_action_max_abs=maximum,
                               new_q_and_native_standardization_used=True, old_R2_q_used=False,
                               PPO_updated=False, fresh_CP_contract=False)


@torch.no_grad()
def roll_constant(policy, starts, residual, horizon, dtype=np.float32):
    states = np.array(starts, dtype=dtype, copy=True)
    outcomes = np.full(len(states), "", dtype=object)
    history, action_history = [states.copy()], []
    first = None
    for step in range(1, horizon + 1):
        active = outcomes == ""
        ids = np.flatnonzero(active)
        if not len(ids): break
        actions = policy(torch.as_tensor(states[ids], dtype=torch.float32)).numpy().reshape(-1)
        full_actions = np.full(len(states), np.nan); full_actions[ids] = actions
        before = states[ids].astype(np.float64)
        nd = before[:, 0] - .05 * before[:, 1]
        nv = np.clip(np.clip(before[:, 1] - .05 * actions.astype(np.float64), 0, 3) + residual, 0, 3)
        events = classify_batch(nd, nv, np.ones(len(ids), dtype=bool), step, horizon)
        states[ids] = np.column_stack((nd, nv))
        outcomes[ids[events != ""]] = events[events != ""]
        history.append(states.copy()); action_history.append(full_actions)
        if first is None and (events == "unsafe").any():
            local = int(np.flatnonzero(events == "unsafe")[0]); row = int(ids[local])
            first = dict(initial_state=np.asarray(starts[row]).astype(float).tolist(),
                         previous_state=before[local].tolist(), unsafe_state=[float(nd[local]), float(nv[local])],
                         unsafe_at_step=step, action=float(actions[local]), residual_each_step=float(residual),
                         initial_index=row)
    outcomes[outcomes == ""] = "timeout"
    return dict(outcomes={name: int((outcomes == name).sum()) for name in (
                    "success", "unsafe", "timeout", "stopped_safe_outside_goal", "out_of_domain")},
                first_counterexample=first), np.stack(history), np.asarray(action_history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--initial-grid-size", type=int, default=20)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() or min(args.initial_grid_size, args.horizon) < 2:
        p.error("New output and valid sizes required")
    torch.set_num_threads(1)
    policy, record, metadata = load_frozen(args.native_dir, args.controller_dir)
    args.output_dir.mkdir(parents=True)
    qref = record["reference_residual"]["same_recipe_radius_mps"]
    qstatic = record["matched"]["internal_audit"]["steps"]["max_abs"]
    modes = dict(zero=0., negative_reference=-qref, positive_reference=qref,
                 negative_static=-qstatic, positive_static=qstatic)
    dd, vv = np.meshgrid(np.linspace(15, 16, args.initial_grid_size),
                         np.linspace(2.5, 3, args.initial_grid_size), indexing="ij")
    starts = np.column_stack((dd.reshape(-1), vv.reshape(-1))).astype(np.float32)
    results = {}
    for name, residual in modes.items():
        result, states, actions = roll_constant(policy, starts, residual, args.horizon)
        witness = result["first_counterexample"]
        if witness:
            row = witness["initial_index"]; end = witness["unsafe_at_step"]
            np.savez_compressed(args.output_dir / (name + "_witness.npz"), states=states[:end+1, row],
                                actions=actions[:end, row], residual_each_step=residual)
            replay, _, _ = roll_constant(policy, np.array([witness["initial_state"]]), residual, args.horizon, np.float64)
            result["float64_dynamics_replay"] = replay
        results[name] = result
        print(name, "residual=%.9f" % residual, result["outcomes"], flush=True)
    blocked = any(results[n]["outcomes"]["unsafe"] for n in ("negative_reference", "positive_reference"))
    metadata["source_sha256"][str(Path(__file__))] = sha256(Path(__file__))
    torch.save(dict(state_dict=policy.state_dict(), metadata=metadata), args.output_dir / "frozen_physical_policy.pt")
    metrics = dict(status="reference_global_box_counterexample_found" if blocked else "no_constant_residual_counterexample_not_a_proof",
                   stage="SBC development feasibility gate; NOT training or CP calibration", metadata=metadata,
                   development_reference_radius_mps=qref, development_static_radius_mps=qstatic,
                   initial_states=len(starts), horizon=args.horizon, modes=results,
                   stop_robust_box_training=blocked, formal_certificate=False,
                   scope="a bad allowed constant residual sequence refutes pathwise-global-box nonincrease SBC with initial<unsafe",
                   not_refuted="other stochastic SBCs, structured residual contracts, or actual image-path safety",
                   PPO_SBC_updated=False)
    for path, digest in metadata["source_sha256"].items():
        if sha256(Path(path)) != digest: raise ValueError("Frozen source changed")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print("[Frozen new R2 PPO SBC feasibility]", metrics["status"], flush=True)
    print("adapter max action gap=%.9g; PPO unchanged" % metadata["adapter_action_max_abs"], flush=True)
    print("reference positive witness:", results["positive_reference"]["first_counterexample"], flush=True)
    print("metrics:", args.output_dir / "metrics.json", flush=True)


if __name__ == "__main__": main()
