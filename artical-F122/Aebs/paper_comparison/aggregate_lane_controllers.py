"""Summarize all registered controller seeds, retaining negative outcomes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .lane_controller_comparison import ARMS,SEEDS,save
from .tasks import RESULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESULTS/'paper_controller_comparison_20261118_v1')
    args=p.parse_args();root=args.root;rows=[];same=[];data={}
    for seed in SEEDS:
        history=json.loads((root/str(seed)/'training_history.json').read_text())
        frozen={arm:json.loads((root/str(seed)/arm/'frozen.json').read_text()) for arm in ARMS}
        same.append(dict(seed=seed,checkpoint_bytes_identical=frozen['spvc']['actor_sha256']==frozen['robust']['actor_sha256'],
             spvc_safety_loss_max=max(row['spvc']['safety'] for row in history),
             robust_safety_loss_max=max(row['robust']['safety'] for row in history)))
        for arm in ARMS:
            folder=root/str(seed)/arm
            current=json.loads((folder/'current_conditional.json').read_text())
            reach=json.loads((folder/'spvt_conditional.json').read_text())
            perf=json.loads((folder/'synthetic_performance.json').read_text())
            data[(seed,arm)]=dict(np.load(folder/'synthetic_per_trajectory.npz'))
            row=dict(seed=seed,arm=arm,current_pass=bool(current.get('passed')),current_seconds=current['elapsed_seconds'],
                     training_seconds=frozen[arm]['training_seconds_excluding_shared_curriculum'],
                     anchor_weight_l2_change=frozen[arm]['actor_weight_l2_change'],spvt_seconds=reach['total_wall_seconds'])
            for entry in reach['contexts'][0]['summary']:row['verified_K'+str(entry['horizon'])]=entry['certified']
            for mode,m in perf['metrics'].items():
                for key,val in m.items():row[mode+'__'+key]=val
            rows.append(row)
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rng=np.random.default_rng(202611188990);ids=rng.integers(2000,size=(5000,2000));paired={}
    for mode in ['iid','persistent','alternating']:
        paired[mode]={}
        for metric in ['mean_abs_e_m','mean_abs_heading_rad','mean_steering_change_rad']:
            key=mode+'__'+metric
            by_seed=np.array([data[(seed,'spvt')][key]-data[(seed,'robust')][key] for seed in SEEDS])
            delta=by_seed.mean(0)
            paired[mode][metric]=dict(spvt_minus_robust=float(delta.mean()),
                per_seed_mean_difference=by_seed.mean(1).tolist(),
                fixed_three_models_paired_trajectory_ci95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),
                spvt_mean=float(np.array([data[(seed,'spvt')][key] for seed in SEEDS]).mean()),
                robust_mean=float(np.array([data[(seed,'robust')][key] for seed in SEEDS]).mean()))
    report=dict(rows=rows,expected_vs_robust_identity=same,spvt_minus_robust_paired=paired,
       training_scope='same robust-trained anchor and fixed B, 240updates; controlled objectives, not original joint synthesis',
       safety_scope='conditional on unchanged synthetic residual support; no new image CP in this experiment',
       expected_vs_worst_interpretation='When both safety penalties remain identically zero, those two arms reduce to identical shared performance/anchor fine-tuning.',
       statistics='2000 paired external scenarios reused across seeds. CI resamples trajectories after averaging the three fixed models; it does not estimate unseen-training-seed uncertainty.',
       timing='240 updates equalizes optimization count, not FLOPs; training time excludes shared hard-buffer refresh. SPVT timings have common300s budgets; differences for identical weights can be timeout scheduling effects.')
    save(root/'summary.json',report)
    print(json.dumps(dict(output=str(root/'summary.json'),rows=len(rows),all_current_pass=all(row['current_pass'] for row in rows),identity=same,paired=paired)),flush=True)


if __name__=='__main__':main()
