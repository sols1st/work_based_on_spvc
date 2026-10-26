"""Finite-horizon, appearance-conditioned SBC candidate on exact image states.

Nonnegative bounded Bernstein coefficients give global nonnegative values. Unsafe
or domain-exit states have value1 and existing safe terminal states value0.
LP minimizes gamma+H*c, not the fraction of nodes with decreasing samples.
Only sampled dynamics are constrained: NO continuous/image/CP certificate.
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.special import comb

from Aebs.conformal.calibrate_controller import classify_batch
from Aebs.dino_latent.benchmark_native_controllers import encode_native
from Aebs.dino_latent.native_sbc_stage import load_frozen
from Aebs.dino_latent.prepare_expanded_data import sha256


def bernstein1d(x, degree):
    x = np.asarray(x, np.float64).reshape(-1, 1)
    if degree < 1 or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
        raise ValueError("Need finite unit coordinates and positive degree")
    k = np.arange(degree+1)
    return comb(degree, k) * x**k * (1-x)**(degree-k)


def split_controls(control, t):
    level = np.asarray(control, np.float64)
    left, right = [level[0]], [level[-1]]
    for _ in range(len(level)-1):
        level = (1-t)*level[:-1]+t*level[1:]
        left.append(level[0]); right.append(level[-1])
    return np.array(left), np.array(right[::-1])


def restriction(degree, low, high):
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid restricted interval")
    left, _ = split_controls(np.eye(degree+1), high)
    _, middle = split_controls(left, low/high)
    return middle


def initial_control_matrix(degree):
    # Original initial set d in[15,16], v in[2.5,3], NOT a recoverability set.
    return np.kron(restriction(degree, 10/11, 1), restriction(degree, 2.5/3, 1))


def barrier_features(states, degree):
    s = np.asarray(states, np.float64)
    if s.ndim != 2 or s.shape[1] != 2 or not np.isfinite(s).all():
        raise ValueError("Need finite physical states [meters,m/s]")
    d, v = s.T
    label = classify_batch(d, v, np.ones(len(s), bool), 0, 400)
    failure = (label == "unsafe") | (label == "out_of_domain") | (v < 0) | (v > 3)
    terminal = (label == "success") | (label == "stopped_safe_outside_goal")
    live = ~failure & ~terminal
    phi = np.zeros((len(s), (degree+1)**2))
    if live.any():
        a = bernstein1d((d[live]-5)/11, degree)
        b = bernstein1d(v[live]/3, degree)
        phi[live] = (a[:, :, None]*b[:, None, :]).reshape(live.sum(), -1)
    return phi, failure.astype(np.float64)


def expected_features(next_states, probabilities, degree):
    """Average outcomes AT EACH state, not states or persistent appearances."""
    states, weights = np.asarray(next_states), np.asarray(probabilities, np.float64)
    if (states.ndim != 3 or states.shape[-1] != 2 or weights.shape != (states.shape[1],)
            or (weights < 0).any() or not np.isclose(weights.sum(), 1)):
        raise ValueError("Need state-by-outcome scenarios and a probability vector")
    p, offset = barrier_features(states.reshape(-1, 2), degree)
    return (p.reshape(*states.shape[:2], -1)*weights[None, :, None]).sum(1), (offset.reshape(states.shape[:2])*weights).sum(1)


def successors(states, action):
    s = np.asarray(states, np.float64)
    a = np.clip(np.asarray(action).reshape(-1), -3, 3)
    return np.column_stack((s[:, 0]-.05*s[:, 1], np.clip(s[:, 1]-.05*a, 0, 3)))


def optimize_candidate(current, next_mean, next_offset, current_offset, degree, horizon, coefficient_upper=20., previous_weights=None):
    count, width = current.shape
    inactive = np.zeros(count, bool)
    upper_next, upper_offset = next_mean.copy(), np.asarray(next_offset).copy()
    if previous_weights is not None:
        # clip(E raw B,1) upper-bounds E clip(raw B,1). For the present
        # deterministic process this is exact; no cross-state averaging.
        inactive = current@previous_weights+current_offset >= 1
        use_one = next_mean@previous_weights+next_offset >= 1
        upper_next[use_one] = 0; upper_offset[use_one] = 1
    active = ~inactive
    drift = upper_next-current
    offset = upper_offset-current_offset
    initial = initial_control_matrix(degree)
    # Variables = Bernstein coefficients, initial upper gamma, positive drift c.
    aub = np.vstack((np.column_stack((drift[active], np.zeros(active.sum()), -np.ones(active.sum()))),
                     np.column_stack((-current[inactive], np.zeros(inactive.sum()), np.zeros(inactive.sum()))),
                     np.column_stack((initial, -np.ones(len(initial)), np.zeros(len(initial))))))
    bub = np.concatenate((-offset[active], current_offset[inactive]-1, np.zeros(len(initial))))
    cost = np.zeros(width+2); cost[-2:] = [1., float(horizon)]
    if not np.isfinite(coefficient_upper) or coefficient_upper < 1:
        raise ValueError("Coefficient cap must include the constant-one feasible baseline")
    fit = linprog(cost, A_ub=aub, b_ub=bub, bounds=[(0, coefficient_upper)]*width+[(0, 1), (0, 1)], method="highs",
                  options=dict(primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8))
    if not fit.success:
        raise RuntimeError("LP failed despite the constant-one feasible baseline: "+fit.message)
    solver_objective = float(cost@fit.x)
    # Never report numerical objective >1 as better than the known feasible
    # constant-one live-state certificate. Preserve the raw objective too.
    selected = fit.x if solver_objective < 1-1e-7 else np.concatenate((np.ones(width), [1., 0.]))
    w, gamma, c = selected[:-2], float(selected[-2]), float(selected[-1])
    residual = float(np.max(aub@selected-bub))
    return w, dict(gamma_initial_upper=gamma, c_expected_drift=c,
                   candidate_failure_budget=gamma+horizon*c,
                   maximum_LP_residual=residual, solver_message=fit.message, solver_objective=solver_objective,
                   trivial_baseline_selected=bool(solver_objective >= 1-1e-7),
                   clipped_barrier=previous_weights is not None,
                   current_value_one_states_discharged=int(inactive.sum()),
                   positive_train_drift_count=int(((drift@w+offset) > 1e-7).sum()),
                   training_states=count, variables=len(fit.x))


@torch.no_grad()
def image_actions(actor, z, speed):
    obs = torch.from_numpy(np.column_stack((z, speed)).astype(np.float32))
    return torch.cat([actor(obs[i:i+8192]).clamp(-3, 3).reshape(-1) for i in range(0, len(obs), 8192)]).numpy()


def labelled_pairs(ids, latent, distances, speed_grid, actor):
    rows = np.repeat(ids, len(speed_grid))
    speed = np.tile(speed_grid, len(ids))
    states = np.column_stack((distances[rows], speed))
    actions = image_actions(actor, latent[rows], speed)
    live = np.array(classify_batch(states[:, 0], states[:, 1], np.ones(len(states), bool), 0, 400)) == ""
    return states[live], successors(states[live], actions[live])


def audit_drift(w, states, next_states, degree, c, clipped=False):
    phi, offset = barrier_features(states, degree)
    pn, on = barrier_features(next_states, degree)
    current, successor = phi@w+offset, pn@w+on
    if clipped:
        current, successor = np.minimum(current, 1), np.minimum(successor, 1)
    drift = successor-current
    return dict(states=len(states), mean_drift=float(drift.mean()), max_drift=float(drift.max()),
                positive_drift_count=int((drift > 1e-7).sum()),
                exceeds_registered_c_count=int((drift > c+1e-7).sum()),
                max_excess_over_c=float(max(0, drift.max()-c)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    p.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--training-distances", type=int, default=128)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--coefficient-upper", type=float, default=20.)
    p.add_argument("--clip-sbc", action="store_true")
    p.add_argument("--refinement-rounds", type=int, default=3)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if (a.output_dir.exists() or min(a.degree, a.training_distances, a.horizon) < 1
            or not np.isfinite(a.coefficient_upper) or a.coefficient_upper < 1 or a.refinement_rounds < 1):
        p.error("New output and positive dimensions required")
    torch.set_num_threads(1); started = time.monotonic()
    policy, _, provenance = load_frozen(a.native_dir, a.controller_dir)
    files = [a.cache_dir/"features.h5", a.cache_dir/"cache_manifest.json", a.representation,
             a.native_dir/"historical_R2.pt", a.controller_dir/"native_ppo.zip", Path(__file__)]
    hashes = {str(path): sha256(path) for path in files}
    manifest = json.loads(files[1].read_text()); rows = manifest["records"]
    if manifest["test_encoded"] != 0 or manifest["feature_sha256"] != hashes[str(files[0])]:
        raise ValueError("Wrong cache/test identity")
    with h5py.File(files[0], "r") as f:
        raw, distances = f["features"][:], f["distance_m"][:]
        if f["filename"].asstr()[:].tolist() != [r["filename"] for r in rows]:
            raise ValueError("Image features/labels misaligned")
    rep = torch.load(a.representation, map_location="cpu")
    native = torch.load(a.native_dir/"historical_R2.pt", map_location="cpu")
    if hashes[str(a.representation)] not in native["config"]["source_sha256"].values():
        raise ValueError("Wrong DINO projection")
    latent = encode_native(raw, native, rep)
    appearances = np.array([str(r["weather"])+"|"+str(r["color"]) for r in rows])
    splits = np.array([r["split"] for r in rows])
    if not set(splits).issubset({"train", "validation"}):
        raise ValueError("Only previously encoded training/validation images allowed")
    speed_grid = np.unique(np.concatenate((np.linspace(0, 3, 17), [.49, .5, .5001, .6, 1., 2.5, 2.75])))
    a.output_dir.mkdir(parents=True)
    config = dict(source_sha256=hashes, controller_identity=provenance,
                  degree_per_coordinate=a.degree, coefficient_upper=a.coefficient_upper,
                  clip_sbc=a.clip_sbc, refinement_rounds=a.refinement_rounds if a.clip_sbc else 0,
                  horizon=a.horizon, speeds_mps=speed_grid.tolist(),
                  training_distances_per_appearance=a.training_distances,
                  basis="tensor Bernstein polynomial, bounded coefficients; NOT the previous neural SBC",
                  environment="weather/color fixed throughout an episode; separate B(x,e), gamma(e), c(e)",
                  conditional_process_model="zero residual of current analytic capture backend; conditional expectation degenerates to one outcome",
                  no_synthetic_noise_added=True, no_nearest_image=True, no_recoverability_set=True,
                  image_errors_in_training="actual frozen image-path actions at exact image distances, not global residual boxes",
                  target="finite-horizon safety against unsafe/domain exit, NOT task completion or infinite-horizon SPVC equivalence",
                  bound_if_all_dynamics_proved="failure <= gamma(e)+H*c(e); appearance-uniform mixture uses average of these budgets",
                  formal_certificate=False, independent_CP=False, PPO_updated=False,
                  global_region_conditions="nonnegative bounded polynomial, optionally capped at1; unsafe/domain-exit B=1 and safe-terminal B=0 by construction; initial upper via Bernstein restriction convex hull",
                  unresolved="dynamics constrained at finite exact-image states; no continuous action-map or new-policy trajectory coverage proof")
    (a.output_dir/"config.json").write_text(json.dumps(config, indent=2, allow_nan=False))
    results = {}; weights = {}
    for appearance in sorted(np.unique(appearances)):
        train_ids = np.flatnonzero((appearances == appearance) & (splits == "train"))
        val_ids = np.flatnonzero((appearances == appearance) & (splits == "validation"))
        values = np.unique(distances[train_ids])
        selected = values[np.unique(np.linspace(0, len(values)-1, min(a.training_distances, len(values))).astype(int))]
        train_ids = train_ids[np.isin(distances[train_ids], selected)]
        st, nt = labelled_pairs(train_ids, latent, distances, speed_grid, policy.actor)
        sv, nv = labelled_pairs(val_ids, latent, distances, speed_grid, policy.actor)
        ph, off = barrier_features(st, a.degree)
        # Interface accepts real conditional scenario probabilities. Here the
        # observed analytic backend has one deterministic process outcome.
        pn, on = expected_features(nt[:, None, :], np.array([1.]), a.degree)
        w, result = optimize_candidate(ph, pn, on, off, a.degree, a.horizon, a.coefficient_upper)
        history = [dict(result)]
        if a.clip_sbc:
            for _ in range(a.refinement_rounds):
                w, result = optimize_candidate(ph, pn, on, off, a.degree, a.horizon, a.coefficient_upper, w)
                history.append(dict(result))
        training_audit = audit_drift(w, st, nt, a.degree, result["c_expected_drift"], a.clip_sbc)
        if training_audit["max_excess_over_c"] > 2e-7:
            raise ValueError("LP surrogate did not imply the actual clipped training condition")
        validation = audit_drift(w, sv, nv, a.degree, result["c_expected_drift"], a.clip_sbc)
        q_actions = policy(torch.as_tensor(sv, dtype=torch.float32)).numpy().reshape(-1)
        proxy = audit_drift(w, sv, successors(sv, q_actions), a.degree, result["c_expected_drift"], a.clip_sbc)
        validation_c = max(0., validation["max_drift"])
        result.update(validation=validation, proxy_validation=proxy, training_audit=training_audit, optimization_history=history,
                      validation_observed_failure_budget=result["gamma_initial_upper"]+a.horizon*validation_c,
                      certified_failure_bound=None, formal_certificate=False)
        weights[appearance] = w; results[appearance] = result
        np.savez_compressed(a.output_dir/("candidate_%02d.npz" % (len(weights)-1)), coefficients=w,
                            appearance=np.array(appearance), training_states=st,
                            image_successors=nt, validation_states=sv, validation_image_successors=nv)
        print("[Finite-horizon SBC] %s gamma=%.6g c=%.6g train_budget=%.6g validation_excess=%d/%d" % (
            appearance, result["gamma_initial_upper"], result["c_expected_drift"],
            result["candidate_failure_budget"], validation["exceeds_registered_c_count"], validation["states"]), flush=True)
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen input changed: "+path)
    output = dict(status="sampled_finite_horizon_SBC_candidate_only", config=config, appearances=results,
                  uniform_appearance_training_budget=float(np.mean([v["candidate_failure_budget"] for v in results.values()])),
                  uniform_appearance_validation_observed_budget=float(np.mean([v["validation_observed_failure_budget"] for v in results.values()])),
                  worst_appearance_training_budget=max(v["candidate_failure_budget"] for v in results.values()),
                  registered_appearances=len(results), all_sources_unchanged=True,
                  certified_safety_probability=None, runtime_seconds=time.monotonic()-started)
    (a.output_dir/"metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    print("[Finite-horizon candidate, NOT a certificate]", flush=True)
    print("uniform appearance train budget=%.9f, validation observed budget=%.9f" % (
        output["uniform_appearance_training_budget"], output["uniform_appearance_validation_observed_budget"]), flush=True)
    print("metrics:", a.output_dir/"metrics.json", flush=True)


if __name__ == "__main__":
    main()
