"""State-dependent physical residual envelope; development comparison only.

VISION-SLS Eq.(31), gamma=0: convex linear-slack fitting in a nonnegative
Bernstein polynomial basis. A whole-trajectory normalized split-CP REFERENCE
then rescales the envelope. Reused old-policy trajectories are NOT independent
current-policy calibration. No PPO/q/DINO/SBC update or safety certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import binom


LOW = np.array([5., 0.])
HIGH = np.array([16., 3.])
PHYSICAL_CAP = .3  # dt=.05, both actions in [-3,3], speed clipping is nonexpansive


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding='utf-8')


def basis(states, degree=2):
    x = np.asarray(states, np.float64)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or not isinstance(degree, int) or not 1 <= degree <= 4):
        raise ValueError('Finite [distance_m,speed_mps] rows and degree 1..4 required')
    if ((x < LOW - 1e-6) | (x > HIGH + 1e-6)).any():
        raise ValueError('Polynomial evaluation outside registered physical domain')
    t = np.clip((x - LOW) / (HIGH - LOW), 0, 1)
    parts = [np.column_stack([math.comb(degree, k) * t[:, j] ** k
                             * (1 - t[:, j]) ** (degree - k)
                             for k in range(degree + 1)]) for j in range(2)]
    return (parts[0][:, :, None] * parts[1][:, None, :]).reshape(len(x), -1)


def trajectory_weights(groups):
    g = np.asarray(groups)
    if g.ndim != 1 or not len(g):
        raise ValueError('Nonempty trajectory IDs required')
    labels, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
    return 1. / (len(labels) * counts[inverse])


def split_trajectories(groups, seed=7):
    labels = np.unique(groups)
    if len(labels) < 15:
        raise ValueError('At least 15 trajectories required for development split')
    order = np.random.default_rng(seed).permutation(labels)
    n = len(order) * 2 // 5
    return dict(fit=np.sort(order[:n]), calibration_reference=np.sort(order[n:2*n]),
                evaluation=np.sort(order[2*n:]))


def fit_envelope(states, residual, groups, degree=2, penalty=20., floor=.001):
    p = basis(states, degree)
    y = np.asarray(residual, np.float64)
    if (y.shape != (len(p),) or not len(y) or not np.isfinite(y).all()
            or (y < 0).any() or (y > PHYSICAL_CAP + 1e-7).any()
            or np.asarray(groups).shape != y.shape
            or not np.isfinite([penalty, floor]).all()
            or penalty <= 1 or not 0 < floor < PHYSICAL_CAP):
        raise ValueError('Invalid residuals, groups or fit settings')
    w = trajectory_weights(groups)
    # min sum w*b + mu*sum w*xi; b+xi >= |residual|, xi>=0.
    a = sparse.hstack((-sparse.csr_matrix(p), -sparse.eye(len(y))), format='csr')
    objective = np.r_[w @ p, penalty * w]
    solution = linprog(objective, A_ub=a, b_ub=-y,
                       bounds=[(floor, PHYSICAL_CAP)] * p.shape[1] + [(0, None)] * len(y),
                       method='highs', options={'primal_feasibility_tolerance': 1e-9,
                                                'dual_feasibility_tolerance': 1e-9})
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError('Envelope LP failed: ' + solution.message)
    coefficients = solution.x[:p.shape[1]]
    prediction = p @ coefficients
    violation = float(np.maximum(y - prediction - solution.x[p.shape[1]:], 0).max())
    if violation > 1e-7:
        raise RuntimeError('Envelope LP primal audit failed')
    return coefficients, dict(solver='scipy.optimize.linprog/HiGHS',
                              objective=float(solution.fun), degree_per_coordinate=degree,
                              coefficients=len(coefficients), slack_variables=len(y),
                              slack_penalty=penalty, ridge_gamma=0., floor_mps=floor,
                              coefficient_upper_mps=PHYSICAL_CAP,
                              maximum_constraint_residual=violation,
                              fit_undercoverage_count=int((y > prediction + 1e-10).sum()),
                              maximum_fitted_slack_mps=float(np.maximum(y-prediction, 0).max()),
                              weighting='equal total weight for each fitting trajectory')


def predict_radius(states, coefficients, degree=2, multiplier=1.):
    x = np.asarray(states, np.float64)
    coefficients = np.asarray(coefficients, np.float64)
    if (x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all()
            or coefficients.shape != ((degree+1)**2,)
            or not np.isfinite(coefficients).all() or (coefficients <= 0).any()
            or not np.isfinite(multiplier) or multiplier < 0):
        raise ValueError('Invalid envelope input/model')
    inside = ((x >= LOW) & (x <= HIGH)).all(1)
    radius = np.full(len(x), PHYSICAL_CAP)
    radius[inside] = np.minimum(PHYSICAL_CAP, multiplier * (basis(x[inside], degree) @ coefficients))
    return radius


def trajectory_maxima(values, groups):
    x, g = np.asarray(values, np.float64), np.asarray(groups)
    if x.ndim != 1 or x.shape != g.shape or not len(x) or not np.isfinite(x).all():
        raise ValueError('Paired finite trajectory scores required')
    labels, inverse = np.unique(g, return_inverse=True)
    result = np.full(len(labels), -np.inf)
    np.maximum.at(result, inverse, x)
    return labels, result


def reference_quantile(scores, alpha=.01):
    s = np.asarray(scores, np.float64)
    if s.ndim != 1 or not len(s) or not np.isfinite(s).all() or not 0 < alpha < 1:
        raise ValueError('Invalid CP-reference scores/alpha')
    rank = int(math.ceil((len(s)+1)*(1-alpha)))
    if rank > len(s):
        raise ValueError('Insufficient trajectories: finite split-CP quantile unavailable')
    return float(np.sort(s)[rank-1]), dict(
        calibration_trajectories=len(s), nominal_marginal_coverage=1-alpha,
        ascending_order_statistic_rank=rank,
        hypothetical_iid_confidence_at_target=float(binom.cdf(rank-1, len(s), 1-alpha)),
        requested_confidence=.99,
        iid_confidence_target_met=bool(binom.cdf(rank-1, len(s), 1-alpha) >= .99),
        actual_fresh_CP_guarantee=False,
        scope='formula reference only: reused off-policy development trajectories')


def summarize(residual, radius, groups):
    r, b, g = np.asarray(residual), np.asarray(radius), np.asarray(groups)
    if (r.ndim != 1 or r.shape != b.shape or r.shape != g.shape or not len(r)
            or not np.isfinite(r).all() or not np.isfinite(b).all()
            or (r < 0).any() or (b < 0).any()):
        raise ValueError('Invalid residual/radius evaluation')
    excess = np.maximum(r-b, 0)
    labels, maxima = trajectory_maxima(excess, g)
    means = [float(b[g == label].mean()) for label in labels]
    return dict(steps=len(r), trajectories=len(labels),
                step_coverage=float((excess <= 1e-10).mean()),
                whole_recorded_trajectory_coverage=float((maxima <= 1e-10).mean()),
                uncovered_steps=int((excess > 1e-10).sum()),
                uncovered_trajectories=int((maxima > 1e-10).sum()),
                mean_radius_mps=float(b.mean()), median_radius_mps=float(np.median(b)),
                trajectory_balanced_mean_radius_mps=float(np.mean(means)),
                p95_radius_mps=float(np.quantile(b, .95)), maximum_radius_mps=float(b.max()),
                minimum_radius_mps=float(b.min()), maximum_exceedance_mps=float(excess.max()),
                empirical_only=True)


def physics_residual(speed, image_action, proxy_action):
    v, ai, aq = [np.asarray(x, np.float64) for x in (speed, image_action, proxy_action)]
    return np.clip(v-.05*np.clip(ai, -3, 3), 0, 3)-np.clip(v-.05*np.clip(aq, -3, 3), 0, 3)


def load_panel(args):
    """Replay frozen actor outputs, validate exact-frame provenance and cache sign."""
    import h5py
    import torch
    from Aebs.dino_latent.models import SafetyProjection
    from Aebs.dino_latent.native_sbc_stage import load_frozen

    torch.set_num_threads(1)
    files = [args.controller_dir/'reference_residual.npz', args.controller_dir/'metrics.json',
             args.controller_dir/'native_ppo.zip', args.native_dir/'historical_R2.pt',
             args.representation, args.trajectories, args.panel_dir/'reference_panel.npz',
             args.panel_dir/'config.json', Path(__file__), Path('Aebs/system/env.py'),
             Path('Aebs/VT/train.py'), Path('Aebs/VT/verify.py')]
    for optional in (args.barrier_dir/'sbc.pt', args.barrier_dir/'metrics.json'):
        if optional.exists(): files.append(optional)
    hashes = {str(path): digest(path) for path in files}
    config = json.loads((args.panel_dir/'config.json').read_text())
    for path in (args.representation, args.trajectories, args.native_dir/'historical_R2.pt'):
        if config['source_sha256'].get(str(path)) != hashes[str(path)]:
            raise ValueError('Source hash differs from frozen comparison: '+str(path))
    policy, record, identity = load_frozen(args.native_dir, args.controller_dir)
    if record['name'] != 'historical_R2': raise ValueError('Unexpected policy')
    data = dict(np.load(args.controller_dir/'reference_residual.npz'))
    panel = dict(np.load(args.panel_dir/'reference_panel.npz'))
    with h5py.File(args.trajectories, 'r') as f:
        if not f.attrs.get('complete'): raise ValueError('Incomplete source trajectories')
        valid = f['valid'][:].astype(bool)
        ep, tt = np.where(valid)
        ids = f['trajectory_id'].asstr()[:]
        weather, color = f['weather'].asstr()[:], f['color'].asstr()[:]
        d, v = f['distance_m'][:], f['speed_mps'][:]
        collection = json.loads(f.attrs['provenance_json'])
    if len(set(ids)) != len(ids): raise ValueError('Duplicate trajectory IDs')
    if (not np.array_equal(panel['trajectory_index'], ep)
            or not np.array_equal(panel['time_index'], tt)
            or not np.array_equal(data['trajectory_index'], ep)):
        raise ValueError('Cached panels are not exact same trajectory/frame rows')
    for key, truth in (('distance_m', d[ep, tt]), ('speed_mps', v[ep, tt])):
        if not np.array_equal(data[key], truth) or not np.array_equal(panel[key], truth):
            raise ValueError('Physical state alignment failed')
    representation = torch.load(args.representation, map_location='cpu')
    native = torch.load(args.native_dir/'historical_R2.pt', map_location='cpu')
    projection = SafetyProjection().eval().requires_grad_(False)
    projection.load_state_dict(representation['projection_state_dict'])
    maximum_image_gap = maximum_q_gap = 0.
    for start in range(0, len(ep), 4096):
        end = start+4096
        with torch.no_grad():
            raw = torch.from_numpy(panel['features'][start:end])
            z = projection((raw-representation['feature_mean'])/representation['feature_std'])
            z = (z-native['latent_mean'])/native['latent_std']
            speed = torch.from_numpy(data['speed_mps'][start:end, None]).float()
            ai = policy.actor(torch.cat((z, speed), 1)).clamp(-3, 3).numpy().reshape(-1)
            aq = policy(torch.from_numpy(np.column_stack((data['distance_m'][start:end],
                                                          data['speed_mps'][start:end]))).float()).numpy().reshape(-1)
        maximum_image_gap = max(maximum_image_gap, float(np.max(abs(ai-data['image_action'][start:end]))))
        maximum_q_gap = max(maximum_q_gap, float(np.max(abs(aq-data['q_action'][start:end]))))
    residual = physics_residual(data['speed_mps'], data['image_action'], data['q_action'])
    residual_gap = float(np.max(abs(residual-data['residual_mps'])))
    if max(maximum_image_gap, maximum_q_gap) > 2e-5 or residual_gap > 1e-7:
        raise ValueError('Cached physical residuals do not replay frozen models')
    states = np.column_stack((data['distance_m'], data['speed_mps']))
    basis(states)  # domain and finite check before writing/fitting
    if not np.isfinite(residual).all() or np.abs(residual).max() > PHYSICAL_CAP+1e-7:
        raise ValueError('Invalid physical residual')
    return states, residual, ep, weather[ep].astype(str), color[ep].astype(str), dict(
        source_sha256=hashes, native_identity=identity,
        collector_controller_sha256=collection.get('controller_sha256'),
        evaluated_controller_sha256=record['controller_sha256'],
        collector_matches_evaluated_policy=collection.get('controller_sha256') == record['controller_sha256'],
        recorded_trajectory_ids=ids.tolist(), frame_time=tt.tolist(),
        replay_image_action_max_abs=maximum_image_gap, replay_q_action_max_abs=maximum_q_gap,
        replay_residual_max_abs_mps=residual_gap,
        collector='old PPO09; current policy evaluated at same recorded states, NOT new closed loop')


def plot_results(output, states, residual, groups, split, methods, coefficients, degree, multiplier):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    evaluation = np.isin(groups, split['evaluation'])
    d, v = np.meshgrid(np.linspace(5, 16, 110), np.linspace(0, 3, 80), indexing='ij')
    grid = np.column_stack((d.ravel(), v.ravel()))
    b = predict_radius(grid, coefficients, degree, multiplier).reshape(d.shape)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    image = ax[0].pcolormesh(d, v, b, shading='auto', cmap='viridis')
    fig.colorbar(image, ax=ax[0], label='Radius [m/s]')
    ax[0].set(title='Fitted map; unobserved regions NOT certified', xlabel='Distance [m]', ylabel='Speed [m/s]')
    rows = np.flatnonzero(evaluation)[::max(1, int(evaluation.sum())//2500)]
    ax[1].scatter(states[rows, 0], np.abs(residual[rows]), s=4, alpha=.3, label='Actual matched residual')
    ax[1].scatter(states[rows, 0], methods['state_cp_reference'][rows], s=4, alpha=.3, label='State radius')
    ax[1].axhline(methods['global_cp_reference'][rows[0]], color='red', label='Same-split global radius')
    ax[1].set(xlabel='Distance [m]', ylabel='Absolute speed residual / radius [m/s]', title='Held-out development trajectories')
    ax[1].legend(fontsize=8)
    fig.savefig(output/'state_radius.png', dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-dir', type=Path, default=Path('results/native_controllers_speed_residual_20261019_v1'))
    p.add_argument('--controller-dir', type=Path, default=Path('results/native_controllers_speed_residual_20261019_v1/historical_R2'))
    p.add_argument('--native-dir', type=Path, default=Path('results/native_representation_dimensions_20261019_v1'))
    p.add_argument('--representation', type=Path, default=Path('results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt'))
    p.add_argument('--trajectories', type=Path, default=Path('results/dino_expanded_v1/20_trajectory_sbc_cp_calibration_v1/trajectories.h5'))
    p.add_argument('--barrier-dir', type=Path, default=Path('results/native_R2_sbc_stage2_legacy_20261019_v1'))
    p.add_argument('--degree', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--slack-penalty', type=float, default=20.)
    p.add_argument('--floor-mps', type=float, default=.001)
    p.add_argument('--alpha', type=float, default=.01)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    if (args.output_dir.exists() or not 0 < args.alpha < 1
            or not 0 < args.floor_mps < PHYSICAL_CAP or args.slack_penalty <= 1):
        p.error('New output directory and valid settings required')
    started = time.monotonic()
    states, residual, groups, weather, color, provenance = load_panel(args)
    split = split_trajectories(groups, args.seed)
    fit = np.isin(groups, split['fit']); cal = np.isin(groups, split['calibration_reference'])
    # Refuse before fitting if finite CP reference cannot be computed.
    reference_quantile(np.zeros(len(split['calibration_reference'])), args.alpha)
    args.output_dir.mkdir(parents=True)
    config = dict(args={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  status='registered_before_envelope_fit',
                  scope='reused off-policy trajectory development; no fresh CP or SBC guarantee',
                  fit_objective='sum(w*b)+mu*sum(w*xi), b+xi>=abs(image_next_v-q_next_v), xi>=0',
                  basis='tensor Bernstein polynomial, positive coefficients; degree2 = 9 coefficients',
                  fit_ridge_gamma=0., state_domain_m_mps=[LOW.tolist(), HIGH.tolist()],
                  radius_physical_cap_mps=PHYSICAL_CAP,
                  outside_domain='return physical .3 m/s cap, never polynomial extrapolation',
                  split={k: v.tolist() for k,v in split.items()}, provenance=provenance,
                  no_nearest_image=True, final_test_used=False, PPO_q_DINO_SBC_updated=False,
                  independent_final_CP=False, safety_certificate=False,
                  score_unit='one entire recorded trajectory, not independently shuffled frames')
    save_json(args.output_dir/'config.json', config)
    print('[State-dependent residual envelope] fitting on %d trajectories / %d exact states' % (len(split['fit']), fit.sum()), flush=True)
    coefficients, fit_info = fit_envelope(states[fit], abs(residual[fit]), groups[fit],
                                          args.degree, args.slack_penalty, args.floor_mps)
    raw = predict_radius(states, coefficients, args.degree)
    _, global_scores = trajectory_maxima(abs(residual[cal]), groups[cal])
    _, state_scores = trajectory_maxima(abs(residual[cal])/raw[cal], groups[cal])
    global_q, cp_info = reference_quantile(global_scores, args.alpha)
    multiplier, _ = reference_quantile(state_scores, args.alpha)
    calibrated = predict_radius(states, coefficients, args.degree, multiplier)
    methods = dict(global_cp_reference=np.full(len(groups), global_q),
                   state_raw_fit=raw, state_cp_reference=calibrated)
    tables = {}
    for part, ids in split.items():
        mask = np.isin(groups, ids)
        tables[part] = {name: summarize(abs(residual[mask]), b[mask], groups[mask]) for name,b in methods.items()}
    evaluation = np.isin(groups, split['evaluation'])
    appearances = np.array([str(w)+'|'+str(c) for w,c in zip(weather,color)])
    group_tables = {}
    for appearance in np.unique(appearances[evaluation]):
        mask = evaluation & (appearances == appearance)
        group_tables[str(appearance)] = {name: summarize(abs(residual[mask]), b[mask], groups[mask]) for name,b in methods.items()}
    misses = {}
    for name, radii in methods.items():
        selected = np.flatnonzero(evaluation & (abs(residual) > radii + 1e-10))
        selected = selected[np.argsort(-(abs(residual[selected])-radii[selected]))]
        misses[name] = [dict(trajectory_index=int(groups[i]),
                            trajectory_id=provenance['recorded_trajectory_ids'][int(groups[i])],
                            recorded_frame=int(provenance['frame_time'][i]),
                            distance_m=float(states[i,0]), speed_mps=float(states[i,1]),
                            appearance=str(appearances[i]), signed_residual_mps=float(residual[i]),
                            radius_mps=float(radii[i]),
                            exceedance_mps=float(abs(residual[i])-radii[i])) for i in selected[:20]]
    cells = []
    for dl, dh in zip(np.linspace(5,16,5)[:-1], np.linspace(5,16,5)[1:]):
        for vl, vh in zip(np.linspace(0,3,5)[:-1], np.linspace(0,3,5)[1:]):
            mask = evaluation & (states[:,0]>=dl) & (states[:,0] < dh if dh<16 else states[:,0]<=dh)
            mask &= (states[:,1]>=vl) & (states[:,1]<vh if vh<3 else states[:,1]<=vh)
            if mask.any(): cells.append(dict(distance_m=[float(dl),float(dh)], speed_mps=[float(vl),float(vh)],
                methods={name: summarize(abs(residual[mask]), b[mask], groups[mask]) for name,b in methods.items()}))
    np.savez_compressed(args.output_dir/'envelope.npz', coefficients=coefficients,
                        degree=args.degree, multiplier=multiplier, floor_mps=args.floor_mps,
                        domain_low=LOW, domain_high=HIGH, physical_cap_mps=PHYSICAL_CAP)
    np.savez_compressed(args.output_dir/'samples.npz', states_m_mps=states, signed_residual_mps=residual,
                        trajectory_index=groups, weather=weather, color=color,
                        **methods)
    save_json(args.output_dir/'calibration_reference.json', dict(
        **cp_info, global_radius_mps=global_q, normalized_multiplier=multiplier,
        global_scores_mps=global_scores.tolist(), normalized_scores=state_scores.tolist()))
    plot_results(args.output_dir, states, residual, groups, split, methods, coefficients, args.degree, multiplier)
    for path, value in provenance['source_sha256'].items():
        if digest(path) != value: raise RuntimeError('Frozen source changed during run: '+path)
    base, candidate = [tables['evaluation'][k] for k in ('global_cp_reference','state_cp_reference')]
    reduction = 1-candidate['trajectory_balanced_mean_radius_mps']/base['trajectory_balanced_mean_radius_mps'] if global_q>0 else None
    maintained = candidate['whole_recorded_trajectory_coverage'] >= base['whole_recorded_trajectory_coverage']-1e-12
    decision = ('development_radius_improved_without_observed_coverage_loss' if reduction is not None and reduction>0 and maintained
                else 'development_tradeoff_or_no_improvement')
    metrics = dict(status=decision, sample_count=len(states), trajectory_count=len(np.unique(groups)),
                   split_counts={k:len(v) for k,v in split.items()}, fit=fit_info,
                   cp_reference=cp_info, global_radius_mps=global_q, normalized_multiplier=multiplier,
                   trajectory_balanced_radius_reduction=reduction, tables=tables,
                   evaluation_appearances=group_tables, evaluation_physical_cells=cells,
                   worst_evaluation_misses=misses,
                   model_sources_unchanged=True, final_test_used=False, new_closed_loop=False,
                   formal_CP_guarantee=False, safety_certificate=False, SBC_trained=False,
                   unresolved=['Reused old-policy panel: development comparison, not current-policy on-policy calibration.',
                               'Finite recorded trajectory coverage does not establish infinite-horizon SBC.',
                               'Marginal score calibration is not state-wise conditional coverage.',
                               'No continuous-domain support guarantee or new-box feasibility verification.'],
                   runtime_seconds=time.monotonic()-started)
    save_json(args.output_dir/'metrics.json', metrics)
    lines = ['[State-dependent speed residual envelope — development only]',
             'trajectories: '+str(metrics['split_counts']),
             'global radius=%.9f m/s; state multiplier=%.6f' % (global_q,multiplier)]
    for name,row in tables['evaluation'].items():
        lines.append('%s: mean radius=%.9f m/s, trajectory coverage=%.2f%%, step coverage=%.2f%%, max exceedance=%.9f m/s' % (
            name,row['trajectory_balanced_mean_radius_mps'],100*row['whole_recorded_trajectory_coverage'],
            100*row['step_coverage'],row['maximum_exceedance_mps']))
    lines.extend(['status: '+decision,
                  '99%% coverage / 99%% confidence target: NOT established; hypothetical IID confidence with this calibration size=%.2f%%.' % (100*cp_info['hypothetical_iid_confidence_at_target']),
                  'No fresh CP / infinite-horizon SBC guarantee; frozen models unchanged.',
                  'metrics: '+str(args.output_dir/'metrics.json')])
    text = '\n'.join(lines)+'\n'
    (args.output_dir/'summary.txt').write_text(text, encoding='utf-8')
    print(text, flush=True)


if __name__ == '__main__':
    main()
