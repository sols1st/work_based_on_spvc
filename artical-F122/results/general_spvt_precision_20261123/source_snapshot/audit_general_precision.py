"""Recompute normal-condition domain-grid completion and CP scores."""
from pathlib import Path
import json
import numpy as np
from .proxy_architecture_compare import write

R=Path('results')


def main():
    aggregate={}
    for task,root in [('spvt',R/'general_spvt_precision_20261123'),('aebs',R/'general_aebs_precision_20261123')]:
        summary=json.loads((root/'summary.json').read_text())
        groups={}
        for key,row in summary.items():
            cal=np.load(root/key/'calibration.npz')
            test=np.load(root/key/'test.npz')
            q=float(np.sort(cal['scores'])[474])
            assert q==row['q_deg' if task=='spvt' else 'q_m']
            assert int((test['scores']<=q).sum())==row['covered']
            assert len(test['scores'])==1000
            if task=='spvt':
                assert int(test['unsafe'].sum())==row['unsafe']
                for tol in [5,10,20]:assert int(((test['tail_max_cm']<=tol)&~test['unsafe']).sum())==row[f'precision{tol}']
                np.testing.assert_allclose(test['rmse_cm'].mean(),row['rmse_cm'])
                np.testing.assert_allclose(test['settling5_seconds'].mean(),row['settling5_seconds'])
                kind=row['model']
                if kind not in ['spvt_pid','spvt_epoch100']:kind=kind.rsplit('_',1)[0]
                group=(row['initial_halfwidth_cm'],row['heading_halfwidth_deg'],kind)
                metrics={k:row[k] for k in ['rmse_cm','q_deg','covered','unsafe','precision5','precision10','precision20','settling5_seconds']}
            else:
                for name,count in row['outcomes'].items():assert int((test['outcomes']==name).sum())==count
                group=(row['distance_start'],row['speed_start'],row['model'],row['actor'])
                metrics=dict(success=row['outcomes']['success'],unsafe=row['outcomes']['unsafe'],q_m=row['q_m'],covered=row['covered'],
                             capped_time_s=row['time_to_goal_capped_seconds']['mean'],trajectory_mae_cm=row['trajectory_mae_m']['mean']*100)
            groups.setdefault(group,[]).append(metrics)
        rows=[dict(group=list(g),fixed_training_seeds=len(v),n_scenarios=1000,
                   metrics={k:float(np.mean([r[k] for r in v])) for k in v[0]}) for g,v in groups.items()]
        write(root/'aggregate.json',rows)
        write(root/'independent_audit.json',dict(arms=len(summary),counts_recomputed=True,cp_n=499,cp_rank=475,test_n=1000,
              scenario_scope='same initial draws paired within and across frozen-model groups'))
        aggregate[task]=rows
    return aggregate


if __name__=='__main__':main()
