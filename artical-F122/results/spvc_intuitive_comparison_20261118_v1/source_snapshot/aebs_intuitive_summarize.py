"""Paired whole-scenario summaries for frozen AEBS accuracy/completion runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

from Aebs.paper_comparison.aebs_intuitive_evaluate import read, write, sha

SEED = 202611189103
METRICS = ('success', 'unsafe', 'stopped_safe_outside_goal', 'out_of_domain', 'timeout',
           'scores', 'episode_mae_m', 'episode_rmse_m', 'network_episode_mae_m',
           'network_peak_error_m', 'lookup_episode_mae_m', 'max_lookup_error_m',
           'terminal_distance_to_goal_m', 'terminal_excess_speed_mps',
           'time_to_goal_capped_seconds', 'terminal_abs_deviation_5p5_m')


def metrics(record):
    return np.column_stack([record[k] if k in record else (record['outcomes'] == k).astype(float)
                            for k in METRICS])


def exact_interval(k, n):
    return [0. if k == 0 else float(beta.ppf(.025, k, n - k + 1)),
            1. if k == n else float(beta.ppf(.975, k + 1, n - k))]


def run(out):
    original = read(out / 'summary.json')
    protocol = read(out / 'protocol.json')
    rows, records, arrays = [], {}, {}
    for key, item in original['arms'].items():
        with np.load(out / key / 'holdout.npz') as z:
            records[key] = dict(z)
        r = records[key]
        arrays[key] = metrics(r)
        successful = r['outcomes'] == 'success'
        n_success = int(successful.sum())
        entry = dict(key=key, library=item['model']['library'], frontend=item['model']['key'],
                     mode=item['model']['mode'], budget=item['model']['budget'],
                     train_seed=item['model']['train_seed'], actor=item['arm']['actor'],
                     n=item['n'], q_m=item['q_m'], covered=item['covered'],
                     **item['outcomes'], mean_peak_m=float(r['scores'].mean()),
                     mean_episode_mae_m=float(r['episode_mae_m'].mean()),
                     mean_episode_rmse_m=float(r['episode_rmse_m'].mean()),
                     mean_network_error_m=float(r['network_episode_mae_m'].mean()),
                     mean_lookup_error_m=float(r['lookup_episode_mae_m'].mean()),
                     mean_terminal_goal_distance_m=float(r['terminal_distance_to_goal_m'].mean()),
                     mean_terminal_excess_speed_mps=float(r['terminal_excess_speed_mps'].mean()),
                     mean_capped_goal_time_s=float(r['time_to_goal_capped_seconds'].mean()),
                     mean_actual_termination_time_s=float((r['steps'] * .05).mean()),
                     success_time_subset_n=n_success,
                     mean_success_time_s=None if n_success == 0 else float((r['steps'][successful] * .05).mean()),
                     mean_auxiliary_distance_5p5_m=float(r['terminal_abs_deviation_5p5_m'].mean()))
        rows.append(entry)
    n = len(next(iter(records.values()))['scores'])
    if n != 1000 or not original['formal']:
        raise ValueError('expected formal1000-scene experiment')
    reference = next(iter(records.values()))
    for r in records.values():
        for field in ('initial', 'appearance'):
            np.testing.assert_array_equal(r[field], reference[field])
    # Multinomial bootstrap counts implement IID resampling of complete episodes.
    weights = np.random.default_rng(SEED).multinomial(n, np.full(n, 1. / n), 4000).astype(float) / n
    pairs = []
    def compare(label, keys_a, keys_b, kind):
        if any(key not in arrays for key in keys_a + keys_b):
            raise ValueError('unavailable preregistered paired arm')
        a = np.stack([arrays[k] for k in keys_a]).mean(axis=0)
        b = np.stack([arrays[k] for k in keys_b]).mean(axis=0)
        diff = b - a
        bootstrap = weights @ diff
        bounds = np.quantile(bootstrap, [.025, .975], axis=0)
        result = dict(label=label, kind=kind, a=keys_a, b=keys_b, difference='B minus A',
                      independent_units=n, frozen_checkpoints_per_side=[len(keys_a), len(keys_b)],
                      scope='paired complete scenarios; if averaged over training seeds, CI is conditional on these fixed checkpoints',
                      metrics={name: dict(mean_a=float(a[:, j].mean()), mean_b=float(b[:, j].mean()),
                                         mean_difference=float(diff[:, j].mean()),
                                         bootstrap95=[float(bounds[0, j]), float(bounds[1, j])],
                                         b_lower_count=int((diff[:, j] < 0).sum()),
                                         b_higher_count=int((diff[:, j] > 0).sum()))
                               for j, name in enumerate(METRICS)})
        if len(keys_a) == len(keys_b) == 1:
            sa, sb = records[keys_a[0]]['outcomes'] == 'success', records[keys_b[0]]['outcomes'] == 'success'
            a_only, b_only = int((sa & ~sb).sum()), int((sb & ~sa).sum())
            result['success_discordance'] = dict(a_only=a_only, b_only=b_only,
                mcnemar_exact_two_sided_p=1. if a_only + b_only == 0 else float(binomtest(a_only, a_only + b_only, .5).pvalue))
        pairs.append(result)
    unique_models = {(r['library'], r['frontend']) for r in rows}
    for library, front in sorted(unique_models):
        compare(library + '__' + front + '__actor_effect',
                [library + '__' + front + '__original_ppo'],
                [library + '__' + front + '__current_robust'], 'controller_only')
    fair_modes = sorted({r['mode'] for r in rows if r['budget'] is not None})
    budgets = sorted({r['budget'] for r in rows if r['budget'] is not None})
    seed_aggregate = []
    for budget in budgets:
        seeds = sorted({r['train_seed'] for r in rows if r['budget'] == budget})
        for mode in fair_modes:
            actor_keys = {}
            for actor in ('original_ppo', 'current_robust'):
                keys = [r['key'] for r in rows if r['budget'] == budget and r['mode'] == mode and r['actor'] == actor]
                if len(keys) != len(seeds):
                    raise ValueError('unbalanced seed aggregation')
                actor_keys[actor] = keys
                values = np.stack([arrays[k] for k in keys]).mean(axis=0)
                seed_aggregate.append(dict(mode=mode, budget=budget, actor=actor, keys=keys, seeds=seeds,
                    independent_scenarios=n, fixed_training_seeds=len(seeds),
                    q_mean_m=float(np.mean([original['arms'][k]['q_m'] for k in keys])),
                    q_median_m=float(np.median([original['arms'][k]['q_m'] for k in keys])),
                    metric_means={name: float(values[:, j].mean()) for j, name in enumerate(METRICS)},
                    metric_training_seed_means={name: [float(arrays[k][:, j].mean()) for k in keys]
                                                for j, name in enumerate(METRICS)}))
            compare('%s_%s_seed_average_actor_effect' % (mode, budget),
                    actor_keys['original_ppo'], actor_keys['current_robust'], 'controller_only_paired_seed_average')
        base = 'gray32_mlp'
        for mode in [m for m in fair_modes if m != base]:
            for actor in ('original_ppo', 'current_robust'):
                all_a, all_b = [], []
                for seed in seeds:
                    # Key names are supplied by the manifest, not inferred from a seed convention.
                    def find(m):
                        matching = [r['key'] for r in rows if r['budget'] == budget and
                                    r['train_seed'] == seed and r['mode'] == m and r['actor'] == actor]
                        if len(matching) != 1:
                            raise ValueError('unpaired training seed')
                        return matching[0]
                    a, b = find(base), find(mode)
                    all_a.append(a); all_b.append(b)
                    compare('%s_vs_%s_%s_%s_%s' % (base, mode, budget, seed, actor), [a], [b], 'frontend_only_same_pixels')
                compare('%s_vs_%s_%s_seed_average_%s' % (base, mode, budget, actor), all_a, all_b,
                        'frontend_only_paired_seed_average')
    for library in ('common', 'full24k'):
        for actor in ('original_ppo', 'current_robust'):
            a = library + '__published_native_head__' + actor
            b = library + '__release_rgb_dino__' + actor
            if a in arrays and b in arrays:
                compare(library + '__release_transfer__' + actor, [a], [b], 'transfer_unequal_training')
    for actor in ('original_ppo', 'current_robust'):
        a = 'common__selected_image_label_oracle__' + actor
        b = 'common__current_state_oracle__' + actor
        if a in arrays and b in arrays:
            compare('oracle_lookup_effect__' + actor, [a], [b], 'diagnostic_oracle')
    write(out / 'paired_comparisons.json', dict(seed=SEED, resamples=4000, method='paired whole-episode bootstrap percentile95',
          multiplicity='exploratory pointwise intervals; no simultaneous or selection-adjusted coverage',
          pairs=pairs, inputs={'summary': sha(out / 'summary.json'), 'protocol': sha(out / 'protocol.json'), 'source': sha(Path(__file__))}))
    write(out / 'seed_aggregate.json', dict(scope='mean of fixed training checkpoints on the same1000 scenarios, not3000 independent episodes', rows=seed_aggregate))
    aggregate_rows = [dict(mode=r['mode'], budget=r['budget'], actor=r['actor'],
                          n_scenarios=r['independent_scenarios'], n_training_seeds=r['fixed_training_seeds'],
                          q_mean_m=r['q_mean_m'], q_median_m=r['q_median_m'], **r['metric_means'])
                      for r in seed_aggregate]
    with (out / 'seed_aggregate.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]))
        writer.writeheader(); writer.writerows(aggregate_rows)
    write(out / 'metric_audit.json', dict(
        safety_vs_success='Original task requires d in [5,6] and v<=.5, not v=0. No metric is called parking error.',
        capped_goal_time='Successful episodes use actual first success time; all other outcomes receive20s. Actual stop time and success-subset mean/n are separate.',
        short_failure='Trajectory peak and frame-mean errors are over active pre-action observations; early failure shortens exposure. Always inspect static common-frame error and outcomes together.',
        accuracy_sources='network mean = |prediction - selected image label|; lookup mean = |selected image label - current physical distance|; total mean = |prediction - current physical distance|. Absolute means need not add.',
        zero_risk='Bootstrap[0,0] for all-zero observed unsafe indicators does not imply zero population risk. Use per-arm exact binomial intervals.',
        goal_midpoint='Distance to5.5m is a new auxiliary preference and cannot establish original SPVC task superiority.',
        cp='New ranks are descriptive/calibrated coverage under this frozen replay law. No new B audit is performed and no original-PPO certificate is implied.',
        coverage_intervals={k: exact_interval(v['covered'], v['n']) for k, v in original['arms'].items()}))
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(arms=len(rows), paired_comparisons=len(pairs), output=str(out))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    run(p.parse_args().output)
