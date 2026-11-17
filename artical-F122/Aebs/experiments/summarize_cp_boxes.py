"""Cross-check box membership and summarize paired blind comparisons."""
import argparse
import csv
import importlib
import json
from pathlib import Path

import numpy as np

from ..conformal.trajectory_boxes import METHODS, widths
from .high_dim_cp_boxes import FEATURES, OUTPUTS, PACKAGES, arms, arrays, digest


def write_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def interval(observations, draws):
    means = observations[draws].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def summarize(results):
    out = results/'high_dim_cp_boxes_20261117'
    out.mkdir(exist_ok=True)
    reports, trials = [], []
    draws = np.random.default_rng(307115).integers(100,size=(5000,100))
    for task in OUTPUTS:
        root = results/(OUTPUTS[task]+'_cp_boxes_20261117_v1')
        check = json.loads((root/'comparison.json').read_text())
        seal = json.loads((root/'all_boxes_frozen.json').read_text())
        backend = importlib.import_module(f'Aebs.{PACKAGES[task]}.cp_boxes')
        by_method = {k: [] for k in METHODS}
        for name, _, arm in arms(task, results, root):
            spec = json.loads((arm/'boxes.json').read_text())
            assert digest(arm/'boxes.json') == seal['contracts'][name]
            assert digest(arm/'holdout.npz') == check['arms'][name]['holdout_sha256']
            f, e = arrays(arm/'holdout.npz', backend, 100)
            fc, ec = arrays(arm/'calibration.npz', backend, 199)
            rows = check['arms'][name]['boxes']
            for k in METHODS:
                item = rows[k]
                cfg = spec['boxes'][k]
                neg, pos = widths(cfg['shape'], fc)
                # Independent split rank, computed from the two signed faces directly.
                c = np.where(ec>=0,ec/pos,-ec/neg).max(axis=(1,2))
                assert cfg['q'] == item['q']
                # Log-linear exp and matrix products can differ a few ulps across platforms.
                np.testing.assert_array_max_ulp(np.float64(np.sort(c)[189]), np.float64(cfg['q']), maxulp=8)
                neg, pos = widths(cfg['shape'], f)
                q = cfg['q']
                scores = np.where(e>=0,e/pos,-e/neg).max(axis=(1,2))
                covered = ((e>=-q*neg)&(e<=q*pos)).all(axis=(1,2))
                np.testing.assert_array_equal(covered,scores<=q)
                volume = (q*(pos+neg)).prod(axis=-1).mean(axis=1)
                ratio = volume/rows['fixed_uniform']['mean_box_volume']
                np.testing.assert_allclose(ratio.mean(),item['volume_relative_to_uniform'])
                assert covered.sum()==item['covered']
                metric = item['audit']['certification_metric']
                by_method[k].append((covered, ratio, metric, item['audit']['passed']))
                for j in range(len(covered)):
                    trials.append(dict(task=task, arm=name, method=k, paired_scenario=j, score=float(scores[j]),
                                       q=q, covered=int(covered[j]), volume_relative_to_uniform=float(ratio[j])))
        for k, values in by_method.items():
            cover = np.stack([c for c,_,_,_ in values])
            ratios = np.stack([v for _,v,_,_ in values])
            cert = np.array([b for _,_,b,_ in values])
            base = np.array([b for _,_,b,_ in by_method['fixed_uniform']])
            cover_ref = np.stack([c for c,_,_,_ in by_method['fixed_uniform']])
            vol_lo, vol_hi = interval(ratios.mean(axis=0), draws)
            cov_lo, cov_hi = interval(100*(cover.astype(float)-cover_ref).mean(axis=0),draws)
            reports.append(dict(task=task, method=k, passed=sum(int(r[3]) for r in values), arms=len(values),
                                volume_ratio=float(ratios.mean()), volume_ratio_lo=vol_lo, volume_ratio_hi=vol_hi,
                                volume_below_uniform_arms=int((ratios.mean(axis=1)<1-1e-12).sum()),
                                mean_covered=float(cover.sum(axis=1).mean()), min_covered=int(cover.sum(axis=1).min()),
                                max_covered=int(cover.sum(axis=1).max()), coverage_delta_lo=cov_lo, coverage_delta_hi=cov_hi,
                                metric=float(cert.mean()), metric_min=float(cert.min()), metric_max=float(cert.max()),
                                metric_delta_pct=float(100*(cert.mean()/base.mean()-1)),
                                metric_above_uniform_arms=int((cert>base+1e-12).sum())))
    write_csv(out/'summary.csv', reports)
    write_csv(out/'blind_trajectories.csv', trials)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    summarize(p.parse_args().results)
