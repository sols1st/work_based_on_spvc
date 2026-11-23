"""Scenario-paired visual-shift effects, retaining every preregistered condition."""
import argparse
import csv
from pathlib import Path

import numpy as np

from Aebs.paper_comparison.aebs_intuitive_evaluate import read, write, sha
from Aebs.paper_comparison.aebs_intuitive_summarize import METRICS, metrics


def run(out):
    report = read(out / 'summary.json')
    protocol = read(out / 'protocol.json')
    nominal_root = Path(protocol['nominal_output'])
    nominal_report = read(nominal_root / 'summary.json')
    records, arrays, rows = {}, {}, []
    for key, item in report['arms'].items():
        with np.load(out / key / 'holdout.npz') as z:
            records[key] = dict(z)
        arrays[key] = metrics(records[key])
        model = item['model']
        rows.append(dict(key=key, condition=model['condition'], mode=model['mode'], budget=model['budget'],
                         train_seed=model['train_seed'], actor=item['arm']['actor'], n=item['n'],
                         nominal_q_m=item['q_m'], inside_nominal_q=item['covered'], **item['outcomes'],
                         **{name + '_mean': float(arrays[key][:, i].mean())
                            for i, name in enumerate(METRICS) if name not in item['outcomes']}))
        nk = item['nominal_key']
        if nk not in arrays:
            with np.load(nominal_root / nk / 'holdout.npz') as z:
                records[nk] = dict(z)
            arrays[nk] = metrics(records[nk])
    reference = next(iter(records.values()))
    for r in records.values():
        np.testing.assert_array_equal(r['initial'], reference['initial'])
        np.testing.assert_array_equal(r['appearance'], reference['appearance'])
    n = len(reference['initial'])
    if n != 1000:
        raise ValueError('expected1000 paired scenarios')
    weights = np.random.default_rng(202611189103).multinomial(n, np.full(n, 1. / n), 4000).astype(float) / n
    pairs = []
    def compare(label, akeys, bkeys, kind):
        a = np.stack([arrays[k] for k in akeys]).mean(axis=0)
        b = np.stack([arrays[k] for k in bkeys]).mean(axis=0)
        diff = b - a
        lo, hi = np.quantile(weights @ diff, [.025, .975], axis=0)
        pairs.append(dict(label=label, kind=kind, a=akeys, b=bkeys, difference='B minus A',
            independent_scenarios=n, checkpoint_count=[len(akeys), len(bkeys)],
            metrics={m: dict(mean_a=float(a[:, i].mean()), mean_b=float(b[:, i].mean()),
                             mean_difference=float(diff[:, i].mean()), bootstrap95=[float(lo[i]), float(hi[i])])
                     for i, m in enumerate(METRICS)}))
    conditions = protocol['conditions']
    modes = sorted({r['mode'] for r in rows})
    seeds = sorted({r['train_seed'] for r in rows})
    aggregate = []
    for condition in conditions:
        for mode in modes:
            actors = {}
            for actor in ('original_ppo', 'current_robust'):
                selected = [r for r in rows if r['condition'] == condition and r['mode'] == mode and r['actor'] == actor]
                selected.sort(key=lambda r: r['train_seed'])
                if len(selected) != 3:
                    raise ValueError('incomplete training seed block')
                keys = [r['key'] for r in selected]
                actors[actor] = keys
                nominal_keys = [report['arms'][key]['nominal_key'] for key in keys]
                values = np.stack([arrays[key] for key in keys]).mean(axis=0)
                aggregate.append(dict(condition=condition, mode=mode, actor=actor, budget=24000,
                    independent_scenarios=n, fixed_training_seeds=3,
                    **{name + '_mean': float(values[:, i].mean()) for i, name in enumerate(METRICS)}))
                compare(condition + '__' + mode + '__' + actor + '__shift_effect_seed_average',
                        nominal_keys, keys, 'visual_shift_effect_seed_average')
                for key, nk in zip(keys, nominal_keys):
                    compare(key + '__shift_effect', [nk], [key], 'visual_shift_effect')
            compare(condition + '__' + mode + '__actor_effect_seed_average',
                    actors['original_ppo'], actors['current_robust'], 'controller_only_seed_average')
            for a, b in zip(actors['original_ppo'], actors['current_robust']):
                compare(a + '__actor_effect', [a], [b], 'controller_only')
        for actor in ('original_ppo', 'current_robust'):
            def keys_for(mode):
                matching = [r for r in rows if r['condition'] == condition and r['mode'] == mode and r['actor'] == actor]
                return [r['key'] for r in sorted(matching, key=lambda r: r['train_seed'])]
            base = keys_for('gray32_mlp')
            for mode in [m for m in modes if m != 'gray32_mlp']:
                alternative = keys_for(mode)
                compare(condition + '__' + mode + '_vs_mlp__' + actor + '__seed_average',
                        base, alternative, 'frontend_only_seed_average')
                for a, b in zip(base, alternative):
                    compare(b + '_vs_mlp', [a], [b], 'frontend_only')
    for name, table in [('summary.csv', rows), ('seed_aggregate.csv', aggregate)]:
        with (out / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    write(out / 'seed_aggregate.json', dict(rows=aggregate,
          scope='average over3 frozen training seeds, sharing1000 scenarios; not3000 independent trials'))
    write(out / 'paired_comparisons.json', dict(pairs=pairs, resamples=4000, seed=202611189103,
          method='paired complete-scenario bootstrap, percentile95, pointwise exploratory intervals',
          warnings=['same1000 scenarios across seeds, conditions, actors, and fronts',
                    'no new shifted-law CP guarantee; nominal q use is descriptive only',
                    'no zero-risk claim from all-zero unsafe bootstrap intervals',
                    'fixed synthetic corruption family is not real unseen-domain capture'],
          sources={'summary': sha(out / 'summary.json'), 'nominal_summary': sha(nominal_root / 'summary.json'),
                   'code': sha(Path(__file__))}))
    print('summarized', len(rows), 'arms', len(pairs), 'paired comparisons', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    run(p.parse_args().output)
