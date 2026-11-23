"""Aggregate frozen-seed results, task settling times and paired-benefit tables."""
from pathlib import Path
import json
import numpy as np
from .proxy_architecture_compare import write

ROOT=Path('results')
A=ROOT/'benefit_cp_aebs_20261122'
S=ROOT/'benefit_native_spvt_20261122'


def main():
    rows=json.loads((A/'summary.json').read_text())
    static=json.loads((A/'static.json').read_text())
    agg=[]
    for condition in sorted({r['condition'] for r in rows.values()}):
        for mode in ('gray32_mlp','gray32_spatial'):
            sm=[v for key,v in static.items() if key.startswith(mode+'_') and key.endswith('_'+condition)]
            for actor in ('original_ppo','current_robust'):
                selected=[v for v in rows.values() if v['condition']==condition and v['mode']==mode and v['actor']==actor]
                agg.append(dict(condition=condition,mode=mode,actor=actor,static_mae_cm=float(np.mean([v['mae_m'] for v in sm])*100),
                    q_m=float(np.mean([v['q_m'] for v in selected])),coverage=float(np.mean([v['covered']/v['n'] for v in selected])),
                    trajectory_mae_cm=float(np.mean([v['trajectory_mae_m']['mean'] for v in selected])*100),
                    success=float(np.mean([v['outcomes']['success'] for v in selected])),n=2000,
                    success_by_seed=[v['outcomes']['success'] for v in selected],
                    unsafe_by_seed=[v['outcomes']['unsafe'] for v in selected]))
    write(A/'aggregate.json',agg)
    source=json.loads((S/'summary.json').read_text())
    groups={}
    for key,row in source.items():
        mode,arm=key.split('__')
        kind=arm if arm in ('spvt_pid','spvt_epoch100') else arm.rsplit('_',1)[0]
        states=np.load(S/key/'test.npz')['states_cm_degrees']
        suffix=np.maximum.accumulate(abs(states[:,::-1,0]),axis=1)[:,::-1]
        reached=(suffix<=5)
        settle=np.where(reached.any(1),reached.argmax(1)*.05,8.)
        row=dict(row,settle5_capped_seconds=float(settle.mean()))
        groups.setdefault((mode,kind),[]).append(row)
    summaries=[]
    for (mode,kind),values in groups.items():
        item=dict(mode=mode,kind=kind,training_seeds=len(values),n_scenarios=2000,
            metrics={k:float(np.mean([v[k] for v in values])) for k in values[0]},
            by_seed=values)
        summaries.append(item)
    write(S/'aggregate.json',summaries)


if __name__=='__main__':main()
