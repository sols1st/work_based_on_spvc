"""Frozen R2: magnitude boxes versus trajectory barrier-risk calibration.

No learning or new CARLA capture. Paired OLD trajectories are off-policy
diagnostics only. New coherent on-policy trajectories use a frozen TRAIN-image
nearest-feature simulator. Any statistical assurance is conditional on that
simulator, not a CARLA/image-path SBC supermartingale certificate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch
from scipy.stats import beta as beta_distribution
from stable_baselines3 import PPO

from Aebs.conformal.calibrate_controller import classify_batch, conformal_summary
from Aebs.dino_latent.benchmark_native_controllers import NativeInterface, actor_actions, encode_native
from Aebs.dino_latent.benchmark_native_representations import validate_split
from Aebs.dino_latent.native_sbc_stage import load_frozen, roll_constant
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.system.env import Aebs
from Aebs.VT.utils import MLP


PRIMARY_SCORE = "positive_variation"
SCORE_NAMES = (PRIMARY_SCORE, "decomposed_budget", "horizon_max_increment", "max_barrier")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def summary(value):
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all(): raise ValueError("Nonempty finite values required")
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), p99=float(np.quantile(x, .99)),
                min=float(x.min()), max=float(x.max()))


def lookup_alignment_gate(interface, max_error_m=.01):
    """Exact covering radius on [5,16], not just errors seen in easy episodes."""
    if not np.isfinite(max_error_m) or max_error_m <= 0:
        raise ValueError("Positive finite image-state alignment tolerance required")
    profiles={}
    for name,pool in interface.pools.items():
        d=np.unique(np.asarray(interface.distances[pool],dtype=np.float64))
        if not len(d) or not np.isfinite(d).all() or d[0]<5 or d[-1]>16:
            raise ValueError("Image labels must be finite and in [5,16]")
        radius=max(float(d[0]-5),float(16-d[-1]),float(np.diff(d).max()/2) if len(d)>1 else 0.)
        profiles[name]=dict(unique_distances=len(d),full_domain_max_error_m=radius)
    return dict(max_allowed_error_m=max_error_m,by_appearance=profiles,
                synchronized_image_eligible=all(r["full_domain_max_error_m"]<=max_error_m for r in profiles.values()),
                scope="observation alignment gate, not CP coverage or a safety relaxation")


def successors(states, actions):
    s = np.asarray(states, dtype=np.float64)
    a = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), -3, 3)
    return np.column_stack((s[:, 0] - .05*s[:, 1], np.clip(s[:, 1] - .05*a, 0, 3))).astype(np.float32)


def stopped_values(raw, states, lower, upper):
    """A monitoring function, not a claim about certified unsafe NN bounds.

    Explicit unsafe value=1 makes unsafe => nonnegative trajectory score
    algebraically true even if the original interval verifier is imperfect.
    The actual frozen neural network is unchanged.
    """
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Invalid fixed barrier normalization")
    s = np.asarray(states)
    result = np.clip((np.asarray(raw, dtype=np.float64).reshape(-1)-lower)/(upper-lower), 0, 1)
    unsafe = (s[:, 0] <= 6) & (s[:, 1] > .5)
    safe_terminal = ~unsafe & (((s[:, 0] >= 5) & (s[:, 0] <= 6) & (s[:, 1] <= .5))
                              | ((s[:, 0] > 6) & (s[:, 1] <= 1e-6)))
    result[safe_terminal] = 0.
    result[unsafe] = 1.
    return result


def path_scores(values, proxy_increments, image_error_increments, horizon):
    """Use ONLY a coherent trajectory (successor t equals current t+1)."""
    b = np.asarray(values, dtype=np.float64)
    dq = np.asarray(proxy_increments, dtype=np.float64)
    de = np.asarray(image_error_increments, dtype=np.float64)
    if (b.ndim != 1 or len(b) < 2 or dq.shape != (len(b)-1,) or de.shape != dq.shape
            or horizon < len(dq) or not np.isfinite(np.concatenate((b, dq, de))).all()
            or not np.allclose(np.diff(b), dq+de, atol=1e-8, rtol=1e-7)):
        raise ValueError("Incoherent trajectory increments")
    inc = np.maximum(np.diff(b), 0.)
    return dict(positive_variation=float(b[0]+inc.sum()-1),
                decomposed_budget=float(b[0]+np.maximum(dq, 0).sum()+np.maximum(de, 0).sum()-1),
                horizon_max_increment=float(b[0]+horizon*inc.max()-1),
                max_barrier=float(b.max()-1))


class FrozenBarrier:
    def __init__(self, path, metrics, scale, normalization=None):
        self.model = MLP([2, 16, 8, 1], activation="tanh", square_output=True).eval()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.requires_grad_(False)
        self.scale = float(scale)
        self.lower = float(metrics["verification"]["domain_lower"] if normalization is None else normalization["lower"])
        self.upper = float(metrics["verification"]["unsafe_lower"] if normalization is None else normalization["upper"])

    @torch.no_grad()
    def raw(self, states):
        s = torch.as_tensor(np.array(states, dtype=np.float32, copy=True))
        s[:, 0] /= self.scale
        return self.model(s).reshape(-1).numpy().astype(np.float64)

    def monitor(self, states):
        return stopped_values(self.raw(states), states, self.lower, self.upper)


def paired_audit(states, ai, aq, barrier, output):
    ni, nq = successors(states, ai), successors(states, aq)
    current, vi, vq = barrier.raw(states), barrier.raw(ni), barrier.raw(nq)
    delta = vi-vq
    old_bad, image_bad = vq-current > 1e-8, vi-current > 1e-8
    normalized = (current-barrier.lower)/(barrier.upper-barrier.lower)
    scopes = dict(all_saved_states=np.ones(len(states), bool),
                  current_below_unsafe=(normalized >= 0) & (normalized < 1),
                  original_style_narrow_band=(normalized > .95) & (normalized < 1))
    tables = {}
    for name, mask in scopes.items():
        tables[name] = dict(states=int(mask.sum()), proxy_increase_count=int((old_bad & mask).sum()),
                           image_increase_count=int((image_bad & mask).sum()),
                           proxy_nonincrease_but_image_increase=int((~old_bad & image_bad & mask).sum()))
    worst = np.argsort(delta)[-10:][::-1]
    np.savez_compressed(output, state_m_mps=states, image_action=ai, proxy_action=aq,
                        raw_B_current=current, raw_B_image_next=vi, raw_B_proxy_next=vq,
                        speed_residual_mps=ni[:, 1].astype(float)-nq[:, 1].astype(float))
    return dict(states=len(states), scope="exact saved current states; NOT new-policy trajectories or a conditional-expectation SBC test",
                additional_process_noise="none: isolate image/proxy difference; legacy VT noise is not reinterpreted as CP error",
                speed_error_mps=summary(abs(ni[:, 1].astype(float)-nq[:, 1].astype(float))),
                raw_barrier_error=summary(delta), positive_barrier_error_fraction=float((delta > 1e-8).mean()),
                scopes=tables, worst_cases=[dict(state=states[i].tolist(), image_action=float(ai[i]),
                proxy_action=float(aq[i]), image_minus_proxy_B=float(delta[i]),
                proxy_increment=float(vq[i]-current[i]), image_increment=float(vi[i]-current[i])) for i in worst])


def run_rollouts(ppo, interface, barrier, n, seed, horizon, output, image=True):
    rng = np.random.default_rng(seed)
    labels = list(interface.pools)
    s = np.column_stack((rng.uniform(15, 16, n), rng.uniform(2.5, 3, n))).astype(np.float32)
    appearance = rng.integers(0, len(labels), n)
    initial = s.copy()
    outcomes = np.full(n, "", dtype=object)
    states = np.full((n, horizon+1, 2), np.nan, np.float32); states[:, 0] = s
    action = np.full((n, horizon), np.nan, np.float32)
    q_action = action.copy(); residual = np.full((n, horizon), np.nan, np.float64)
    b = np.full((n, horizon+1), np.nan); b[:, 0] = barrier.monitor(s)
    dq = np.full((n, horizon), np.nan); de = dq.copy()
    lookup = dq.copy(); valid = np.zeros((n, horizon), bool)
    for step in range(horizon):
        ids = np.flatnonzero(outcomes == "")
        if not len(ids): break
        current = s[ids]
        zq = interface.proxy(current[:, 0])
        aq = actor_actions(ppo, zq, current[:, 1])
        if image:
            zi = np.empty_like(zq)
            for j, name in enumerate(labels):
                take = appearance[ids] == j
                if take.any():
                    zi[take], errors, _ = interface.nearest(current[take, 0], name)
                    lookup[ids[take], step] = errors
            ai = actor_actions(ppo, zi, current[:, 1])
        else:
            ai = aq.copy(); lookup[ids, step] = 0
        ni, nq = successors(current, ai), successors(current, aq)
        bi, bq = barrier.monitor(ni), barrier.monitor(nq)
        dq[ids, step] = bq-b[ids, step]
        de[ids, step] = bi-bq
        residual[ids, step] = ni[:, 1].astype(float)-nq[:, 1].astype(float)
        states[ids, step+1] = ni; b[ids, step+1] = bi
        action[ids, step] = ai; q_action[ids, step] = aq; valid[ids, step] = True
        s[ids] = ni
        events = classify_batch(ni[:, 0], ni[:, 1], np.ones(len(ids), bool), step+1, horizon)
        outcomes[ids[events != ""]] = events[events != ""]
    if (outcomes == "").any(): raise ValueError("A rollout has no registered terminal outcome")
    scores = {name: np.empty(n) for name in SCORE_NAMES}
    speed_scores, mismatch_budgets = np.empty(n), np.empty(n)
    for i in range(n):
        length = int(valid[i].sum())
        row = path_scores(b[i, :length+1], dq[i, :length], de[i, :length], horizon)
        for name in SCORE_NAMES: scores[name][i] = row[name]
        if outcomes[i] == "unsafe" and any(value < -1e-10 for value in row.values()):
            raise ValueError("Unsafe trajectory falsely has a negative safety score")
        speed_scores[i] = np.max(abs(residual[i, :length]))
        mismatch_budgets[i] = np.maximum(de[i, :length], 0).sum()
    np.savez_compressed(output, states_m_mps=states, image_action=action, proxy_action=q_action,
                        residual_mps=residual, valid=valid, stopped_barrier=b,
                        proxy_increments=dq, mismatch_increments=de, lookup_error_m=lookup,
                        appearance_index=appearance, appearance_labels=np.array(labels), outcomes=outcomes.astype(str),
                        speed_scores=speed_scores, mismatch_budgets=mismatch_budgets, **scores)
    counts = dict(Counter(outcomes.tolist()))
    return dict(episodes=n, seed=seed, outcomes=counts, mean_steps=float(valid.sum(1).mean()),
                initial_state_sha256=__import__("hashlib").sha256(initial.tobytes()).hexdigest(),
                by_appearance={label: dict(Counter(outcomes[appearance == j].tolist())) for j, label in enumerate(labels)},
                speed_error_mps=summary(residual[valid]), absolute_speed_error_mps=summary(abs(residual[valid])),
                trajectory_max_speed_error_mps=summary(speed_scores), lookup_error_m=summary(lookup[valid]),
                positive_barrier_mismatch_budget=summary(mismatch_budgets),
                scores={name: summary(values) for name, values in scores.items()}), scores, speed_scores


def calibrate_and_evaluate(calibration, evaluation, epsilon, beta):
    output = {}
    for name in SCORE_NAMES:
        cp = conformal_summary(calibration[name], epsilon, beta)
        q = cp["q_hat"]
        output[name] = dict(tolerance=cp, q_strictly_negative=bool(q < 0),
                            evaluation_coverage=float((evaluation[name] <= q).mean()),
                            evaluation_negative_score_fraction=float((evaluation[name] < 0).mean()),
                            individual_scope="fixed library simulator ONLY; primary prespecified; no simultaneous four-method claim")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--barrier-dir", type=Path, default=Path("results/native_R2_sbc_stage2_legacy_20261019_v1"))
    p.add_argument("--reference-panel", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/reference_panel.npz"))
    p.add_argument("--risk-monitor-dir", type=Path, default=None,
                   help="Optional separately trained trajectory monitor; NOT the original legacy SBC")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--max-image-state-gap-m",type=float,default=.01)
    p.add_argument("--allow-approximate-library",action="store_true",
                   help="Explicitly reproduce the old approximate library SIMULATOR, never a synchronized image-path evaluation")
    p.add_argument("--development-episodes", type=int, default=200)
    p.add_argument("--calibration-episodes", type=int, default=459)
    p.add_argument("--evaluation-episodes", type=int, default=1000)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--epsilon", type=float, default=.01)
    p.add_argument("--beta", type=float, default=.01)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or a.seed_offset < 0 or min(a.development_episodes, a.calibration_episodes, a.evaluation_episodes, a.horizon) < 1:
        p.error("New directory and positive rollout counts required")
    # Fail on insufficient sample size BEFORE generating or inspecting outcomes.
    conformal_summary(np.zeros(a.calibration_episodes), a.epsilon, a.beta)
    torch.set_num_threads(1)
    physical, controller_record, metadata = load_frozen(a.native_dir, a.controller_dir)
    barrier_path = a.barrier_dir / "sbc.pt"
    barrier_metrics = json.loads((a.barrier_dir / "metrics.json").read_text())
    if (barrier_metrics["saved_barrier_sha256"] != sha256(barrier_path)
            or barrier_metrics["checkpoint_matches_verification"] is not True
            or barrier_metrics["config"]["provenance"]["source_sha256"] != metadata["source_sha256"]):
        raise ValueError("The SBC is not bound to this frozen native controller")
    paths = [a.representation, a.native_dir/"historical_R2.pt", a.native_dir/"split.json",
             a.controller_dir/"native_ppo.zip", a.controller_dir/"metrics.json", barrier_path,
             a.barrier_dir/"metrics.json", a.cache_dir/"cache_manifest.json", a.cache_dir/"features.h5",
             a.reference_panel, Path(__file__)]
    hashes = {str(path): sha256(path) for path in paths}
    rep = torch.load(a.representation, map_location="cpu")
    ck = torch.load(a.native_dir/"historical_R2.pt", map_location="cpu")
    if hashes[str(a.representation)] not in ck["config"]["source_sha256"].values():
        raise ValueError("Native coordinates do not belong to the requested R2 projection")
    backend_hashes = barrier_metrics["config"]["provenance"]["backend_sha256"]
    for path, digest in backend_hashes.items():
        if sha256(Path(path)) != digest: raise ValueError("Original backend changed since SBC training")
        hashes[path] = digest
    manifest = json.loads((a.cache_dir/"cache_manifest.json").read_text())
    rows = manifest["records"]
    if manifest.get("test_encoded") != 0 or manifest["feature_sha256"] != hashes[str(a.cache_dir/"features.h5")]:
        raise ValueError("Cache fingerprint or test guard failed")
    split = validate_split(json.loads((a.native_dir/"split.json").read_text()), rows)
    with h5py.File(a.cache_dir/"features.h5", "r") as stream:
        raw, distances = stream["features"][:], stream["distance_m"][:].astype(np.float32)
        if (not stream.attrs.get("complete", False)
                or stream["filename"].asstr()[:].tolist() != [r["filename"] for r in rows]
                or not np.allclose(distances, [r["distance_m"] for r in rows], atol=2e-6, rtol=0)):
            raise ValueError("Incomplete/misaligned feature cache")
    latent = encode_native(raw, ck, rep)
    interface = NativeInterface(ck, latent, distances, rows, split["train"])
    alignment=lookup_alignment_gate(interface,a.max_image_state_gap_m)
    if not alignment["synchronized_image_eligible"] and not a.allow_approximate_library:
        p.error("Distance-block split leaves missing image intervals: this is NOT synchronized image-path evaluation. "
                "Use Aebs.dino_latent.audit_image_lookup_accuracy for exact-state accuracy. "
                "--allow-approximate-library is only for explicitly labelled simulator reproduction.")
    ppo = PPO.load(a.controller_dir/"native_ppo.zip", device="cpu")
    ppo.policy.set_training_mode(False); ppo.policy.requires_grad_(False)
    env = Aebs(.05)
    normalization = None
    if a.risk_monitor_dir is not None:
        monitor_metrics_path = a.risk_monitor_dir/"metrics.json"
        monitor_metrics = json.loads(monitor_metrics_path.read_text())
        monitor_path = a.risk_monitor_dir/"risk_monitor.pt"
        if (monitor_metrics["experiment"] != "native_R2_development_occupancy_risk_monitor_v1"
                or monitor_metrics["monitor_sha256"] != sha256(monitor_path)
                or monitor_metrics["controller_sha256"] != hashes[str(a.controller_dir/"native_ppo.zip")]
                or monitor_metrics["native_checkpoint_sha256"] != hashes[str(a.native_dir/"historical_R2.pt")]):
            raise ValueError("Wrong frozen trajectory risk monitor")
        hashes[str(monitor_metrics_path)], hashes[str(monitor_path)] = sha256(monitor_metrics_path), sha256(monitor_path)
        normalization = monitor_metrics["normalization"]
        barrier_path = monitor_path
    barrier = FrozenBarrier(barrier_path, barrier_metrics, env.std1, normalization)
    a.output_dir.mkdir(parents=True)
    config = dict(experiment="frozen_native_R2_trajectory_barrier_CP_comparison_v1",
                  primary_score=PRIMARY_SCORE, comparison_scores=list(SCORE_NAMES), epsilon=a.epsilon, beta=a.beta,
                  horizon=a.horizon, registered_before_rollouts=True,
                  rollout_counts=dict(development=a.development_episodes, calibration=a.calibration_episodes, evaluation=a.evaluation_episodes),
                  seeds=dict(development=7101+a.seed_offset, calibration=7102+a.seed_offset, evaluation=7103+a.seed_offset),
                  risk_monitor="original frozen legacy SBC" if normalization is None else "separate development occupancy monitor, original SBC preserved",
                  initial_distribution="independent distance U[15,16] and speed U[2.5,3]",
                  appearance_distribution="one of 12 train-library weather/color pools uniformly, held fixed per episode",
                  simulator="analytic clipped AEBS; float32 physical states; nearest TRAIN image feature by exact appearance",
                  new_RGB_captured=False, new_CARLA_rollout=False, RGB_test_used=False, model_training=False,
                  additional_process_noise=False,
                  image_state_alignment_gate=alignment,
                  approximate_library_explicitly_allowed=a.allow_approximate_library,
                  barrier_normalization=dict(distance_scale=barrier.scale, lower=barrier.lower, upper=barrier.upper,
                                            function="clip((raw_B-lower)/(upper-lower),0,1); unsafe=1; safe terminal=0"),
                  unsafe="d<=6 and v>0.5; all scores include successor of final recorded action",
                  guarantee_scope="independent trajectories in this frozen image-library simulator only; NOT real image/CARLA safety or classical SBC conditions",
                  source_sha256=hashes)
    save_json(a.output_dir/"config.json", config)
    started = time.monotonic()
    panel = np.load(a.reference_panel)
    s = np.column_stack((panel["distance_m"], panel["speed_mps"])).astype(np.float32)
    zi = encode_native(panel["features"], ck, rep)
    audit = paired_audit(s, actor_actions(ppo, zi, s[:, 1]),
                         actor_actions(ppo, interface.proxy(s[:, 0]), s[:, 1]), barrier,
                         a.output_dir/"old_trajectory_paired.npz")
    static = np.load(a.controller_dir/"internal_audit_matched.npz")
    static_s = np.column_stack((static["distance_m"], static["speed_mps"])).astype(np.float32)
    audit_static = paired_audit(static_s, static["image_action"], static["q_action"], barrier,
                                a.output_dir/"static_paired.npz")
    save_json(a.output_dir/"paired_diagnostic.json", dict(old_trajectory=audit, exact_static_internal_audit=audit_static))
    results, all_scores, speeds = {}, {}, {}
    for role, seed in config["seeds"].items():
        n = config["rollout_counts"][role]
        result, scores, speed = run_rollouts(ppo, interface, barrier, n, seed, a.horizon,
                                            a.output_dir/(role+"_rollouts.npz"))
        results[role], all_scores[role], speeds[role] = result, scores, speed
        print("[Library trajectories]", role, result["outcomes"], "primary", result["scores"][PRIMARY_SCORE], flush=True)
    proxy_result, _, _ = run_rollouts(ppo, interface, barrier, a.evaluation_episodes, config["seeds"]["evaluation"], a.horizon,
                                     a.output_dir/"evaluation_proxy_rollouts.npz", image=False)
    if proxy_result["initial_state_sha256"] != results["evaluation"]["initial_state_sha256"]:
        raise ValueError("Image/proxy evaluation initial states differ")
    cp = calibrate_and_evaluate(all_scores["calibration"], all_scores["evaluation"], a.epsilon, a.beta)
    speed_cp = conformal_summary(speeds["calibration"], a.epsilon, a.beta)
    dd, vv = np.meshgrid(np.linspace(15, 16, 20), np.linspace(2.5, 3, 20), indexing="ij")
    starts = np.column_stack((dd.reshape(-1), vv.reshape(-1))).astype(np.float32)
    strict = {}
    for name, radius in (("old_reference_radius", controller_record["reference_residual"]["same_recipe_radius_mps"]),
                         ("new_library_radius", speed_cp["q_hat"])):
        result, _, _ = roll_constant(physical, starts, radius, a.horizon)
        strict[name] = dict(radius_mps=radius, **result)
    count = results["evaluation"]["outcomes"].get("unsafe", 0); total = a.evaluation_episodes
    risk_upper = float(beta_distribution.ppf(1-a.beta, count+1, total-count)) if count < total else 1.
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest: raise ValueError("Frozen source changed")
    primary_pass = cp[PRIMARY_SCORE]["q_strictly_negative"]
    output = dict(config=config, status="library_trajectory_risk_calibration_pass" if primary_pass else "library_trajectory_risk_calibration_not_passed",
                  paired=dict(old_trajectory=audit, exact_static_internal_audit=audit_static), rollouts=results,
                  matched_initial_proxy_evaluation=proxy_result, trajectory_score_calibration=cp,
                  speed_radius_calibration_library_only=speed_cp, global_box_stress=strict,
                  evaluation_collision_probability_upper_bound=dict(value=risk_upper, confidence=1-a.beta,
                      unsafe=count, episodes=total, method="one-sided exact Clopper-Pearson; frozen library simulator only"),
                  models_bytewise_unchanged=True, original_backend_modified=False, new_CARLA_data=False,
                  real_image_CP_safety_guarantee=False, classical_SBC_conditions_verified=False,
                  runtime_seconds=time.monotonic()-started)
    save_json(a.output_dir/"metrics.json", output)
    print("[Trajectory barrier CP comparison]", output["status"], flush=True)
    for name, row in cp.items():
        print(name, "q=%.9f evaluation_coverage=%.4f" % (row["tolerance"]["q_hat"], row["evaluation_coverage"]), flush=True)
    print("global boxes:", {k:v["outcomes"] for k,v in strict.items()}, flush=True)
    print("metrics:", a.output_dir/"metrics.json", flush=True)


if __name__ == "__main__": main()
