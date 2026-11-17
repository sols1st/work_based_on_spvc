"""Audit matched high-dimensional vision proxies using saved per-trajectory data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..curve_braking import plant as brake
from ..curve_braking.verify import verify_checkpoint
from ..overhead_crane import certificate as crane_cert, plant as crane
from ..visual_hover import certificate as hover_cert, plant as hover


TASKS = {
    'braking': ('curve_braking_proxy_comparison_20261117_v1', [20261117501, 20261117502, 20261117503]),
    'hover': ('hover_proxy_comparison_20261117_v1', [617201, 617202, 617203]),
    'crane': ('crane_proxy_comparison_20261117_v1', [817731, 817732, 817733])
}


def read(path):
    return json.loads(path.read_text())


def verifier(task, results):
    if task == 'braking':
        checkpoint = results / 'curve_braking_5d_20261117/barrier.pt'
        return lambda q: verify_checkpoint(checkpoint, q)['passed']
    if task == 'hover':
        design = read(results / 'visual_hover_6d_20261117_v1/evaluation/synthesis.json')['selected']
        return lambda q: hover_cert.verify(design, q)['passed']
    gains = read(results / 'overhead_crane_8d_20261117_v1/evaluation/synthesis.json')['selected']['gains']
    return lambda q: crane_cert.verify(gains, q, exact=True)['passed']


def radius_budget(check):
    # All three checks are monotone in the nonnegative radius for a fixed backend.
    lo, hi = 0., 1.
    if not check(lo) or check(hi):
        raise ValueError('expected [0, 1] to bracket the certificate boundary')
    for _ in range(40):
        mid = (lo + hi) / 2
        if check(mid):
            lo = mid
        else:
            hi = mid
    return dict(largest_tested_pass=lo, smallest_tested_failure=hi)


def endpoints(task, data):
    states = data['states']
    if task == 'braking':
        lane, gap = brake.margins(states, data['kappas'][:, None])
        safe = ((lane > 0) & (gap > 0)).all(axis=1)
        distance = states[:, -1, 2]
        goals = ((states[:, -1, 3] == 0) & (distance >= brake.TARGET[0])
                 & (distance <= brake.TARGET[1]) & safe)
        return safe, goals, np.abs(distance - np.mean(brake.TARGET))
    if task == 'hover':
        return hover.outcomes(states)
    metrics = [crane.metrics(s[None]) for s in states]
    return (np.array([m['safe'] for m in metrics], dtype=bool),
            np.array([m['settled'] for m in metrics], dtype=bool),
            np.array([m['mean_final_load_m'] for m in metrics]))


def scenario_data(data):
    result = {'initial_states': data['states'][:, 0]}
    result.update({k: data[k] for k in ('kappas', 'winds', 'noise', 'lights') if k in data})
    return result


def paired_interval(per_seed_changes, draws):
    scenario_means = np.stack(per_seed_changes).mean(axis=0)
    low, high = np.quantile(scenario_means[draws].mean(axis=1), [.025, .975])
    return dict(mean=float(scenario_means.mean()), low=float(low), high=float(high))


def analyze(task, results, settings, trajectory_rows):
    folder, seeds = TASKS[task]
    check = verifier(task, results)
    budget = radius_budget(check)
    pairs, scenarios = [], {}
    differences = {'mean_peak_error': [], 'target_rate': [], 'terminal_distance_m': []}
    for seed in seeds:
        metrics, holdouts = {}, {}
        for kind in ('linear', 'neural'):
            name = f'{kind}_{seed}' if task == 'crane' else f'{seed}_{kind}'
            arm = results / folder / name
            scores, raw_scores = {}, {}
            for split, n in [('calibration', 199), ('holdout', 100)]:
                with np.load(arm / f'{split}.npz', allow_pickle=False) as data:
                    errors = data['errors']
                    expected = (n, settings['horizon'], settings['visual_dimensions'])
                    if errors.shape != expected or not np.isfinite(errors).all():
                        raise ValueError(f'{arm} {split}: invalid sensor error arrays')
                    scores[split] = np.abs(errors).max(axis=(1, 2))
                    raw_key = {'braking': 'raw_latent_errors', 'hover': 'raw', 'crane': 'latent_errors'}[task]
                    raw = data[raw_key]
                    if raw.shape != errors.shape or not np.isfinite(raw).all():
                        raise ValueError(f'{arm} {split}: invalid latent residual arrays')
                    raw_scores[split] = np.abs(raw).max(axis=(1, 2))
                    np.testing.assert_allclose(scores[split], data['scores'], rtol=0, atol=1e-10)
                    external = scenario_data(data)
                    if split not in scenarios:
                        scenarios[split] = external
                    if set(scenarios[split]) != set(external):
                        raise ValueError(f'{arm} {split}: incomplete scenario binding')
                    for key in external:
                        np.testing.assert_array_equal(external[key], scenarios[split][key])
                    safe, goals, final_m = endpoints(task, data)
                    for j in range(n):
                        trajectory_rows.append(dict(
                            task=task, train_seed=seed, proxy=kind, split=split, scenario=j,
                            peak_decoded=float(scores[split][j]), peak_latent=float(raw_scores[split][j]),
                            safe=int(safe[j]), goal=int(goals[j]), final_distance_m=float(final_m[j])))
            raw_q = float(np.sort(raw_scores['calibration'])[189])
            q = float(np.sort(scores['calibration'])[189])
            certificate = read(arm / 'certificate.json')
            contract = certificate if task == 'crane' else read(arm / 'contract.json')
            q_saved = contract.get('q', contract.get('radius'))
            if q_saved != q:
                raise ValueError(f'{arm}: recorded q={q_saved} differs from rank 190: {q}')
            passed = bool(check(q))
            recorded = certificate['audit'] if task == 'crane' else certificate
            if passed != recorded['passed']:
                raise ValueError(f'{arm}: recorded certificate disagrees with re-verification')
            peak = scores['holdout']
            metrics[kind] = dict(q=q, q_latent=raw_q, covered_latent=int((raw_scores['holdout'] <= raw_q).sum()),
                                 certificate=passed, q_fraction_of_budget=q/budget['largest_tested_pass'],
                                 covered=int((peak <= q).sum()), safe=int(safe.sum()), goals=int(goals.sum()),
                                 mean_peak=float(peak.mean()), mean_terminal_m=float(np.mean(final_m)))
            holdouts[kind] = dict(mean_peak_error=peak, target_rate=goals.astype(float), terminal_distance_m=final_m)
        for key in differences:
            differences[key].append(holdouts['neural'][key] - holdouts['linear'][key])
        pairs.append(dict(seed=seed, **metrics))
    draws = np.random.default_rng(725116).integers(100, size=(10000, 100))
    return dict(radius_budget=budget, seeds=pairs,
                neural_minus_linear={k: paired_interval(v, draws) for k, v in differences.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    args = parser.parse_args()
    out = args.results / 'high_dim_proxy_comparison_20261117'
    protocol = read(out / 'protocol.json')
    trajectory_rows = []
    summary = {task: analyze(task, args.results, protocol['tasks'][task], trajectory_rows) for task in TASKS}
    with (out / 'per_trajectory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trajectory_rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(trajectory_rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
