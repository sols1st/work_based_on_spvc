"""Supplement tracking with prespecified paired physical steering-effort summaries."""
import argparse
from pathlib import Path

import numpy as np

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.robust_trajectory_cp import write_json
from .train import ARMS,read


def run(root,register):
    path=root/'steering_protocol.json'
    if register:
        if path.exists() or any((root/arm/'holdout_batches').exists() for arm in ARMS):
            raise ValueError('supplement must be specified before measuring any blind trajectory')
        write_json(path,dict(analysis=sha256(Path(__file__)),models={arm:sha256(root/arm/'frozen.json') for arm in ARMS},
                    average='all controlled steps; compute each episode first and then average equally',
                    measures=['mean absolute steering change in radians per control step',
                              'RMS steering radians per trajectory','fraction of steps saturated',
                              '95th percentile across episodes of maximum absolute lateral distance'],
                    strata=['all','absolute curvature equals .02'],pairwise='episode differences vs uniform arm',
                    ci='paired percentile bootstrap 4000 rounds; individual nominal 95% intervals',
                    seed=202611178306))
        return
    protocol=read(path)
    assert protocol['analysis']==sha256(Path(__file__))
    target=root/'steering_metrics.json'
    if target.exists():raise FileExistsError(target)
    saved={}
    for arm in ARMS:
        assert protocol['models'][arm]==sha256(root/arm/'frozen.json')
        with np.load(root/arm/'holdout.npz') as z:
            normalized=z['actions']
            angle=normalized*core.PHI_MAX
            saved[arm]=dict(kappas=z['kappas'],init=z['states'][:,0],
                            change_rad=np.abs(np.diff(angle,axis=1)).mean(1),
                            rms_rad=np.sqrt(np.square(angle).mean(1)),
                            saturation_fraction=(abs(normalized)>=1).mean(1),
                            peak_e_m=np.abs(z['states'][:,1:,0]).max(1))
    ref=saved['uniform']
    report={}
    for group,mask in [('all',np.ones(len(ref['kappas']),bool)),('higher_curvature',abs(ref['kappas'])==.02)]:
        rec={}
        for arm,row in saved.items():
            np.testing.assert_array_equal(ref['kappas'],row['kappas'])
            np.testing.assert_array_equal(ref['init'],row['init'])
            metrics={k:float(row[k][mask].mean()) for k in ['change_rad','rms_rad','saturation_fraction']}
            metrics['q95_peak_e_m']=float(np.quantile(row['peak_e_m'][mask],.95))
            if arm!='uniform':
                delta=(row['change_rad']-ref['change_rad'])[mask]
                rng=np.random.default_rng(protocol['seed'])
                draws=rng.choice(delta,size=(4000,len(delta)),replace=True).mean(1)
                metrics['paired_change_minus_uniform_rad']=dict(delta=float(delta.mean()),
                       ci95=np.quantile(draws,[.025,.975]).tolist())
            rec[arm]=metrics
        report[group]=dict(n=int(mask.sum()),arms=rec)
    write_json(target,dict(protocol=sha256(path),results=report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    run(args.root,args.register)
