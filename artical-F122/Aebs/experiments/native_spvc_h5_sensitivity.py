"""Post hoc H5 sensitivity to images with a spatially constant bottom quarter.

Use fresh paired episode seeds. Preserve the primary results on all 80 images.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.experiments.native_spvc_camera_cross import (
    ACTORS, GRAY, H5, Models, collect, write,
)
from Aebs.vision_sls.run import sha256


@torch.inference_mode()
def run(source):
    out=source/'h5_sensitivity'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    files=[Path(__file__),Path(__file__).with_name('native_spvc_camera_cross.py'),
           source/'protocol.json',source/'frame_estimates.npz',H5,GRAY/'splits.npz']
    write(out/'protocol.json',dict(inputs={str(f):sha256(f) for f in files},
        exploratory=True,seeds=[202611170518,202611170519],calibration_n=199,holdout_n=500,rank=190,
        rule='exclude test images whose 8 bottom rows are exactly constant across all 256 pixel values',
        reason='chosen after inspecting worst native H5 case; same filter for both methods'))
    models=Models()
    with h5py.File(H5) as h:
        flag=np.ptp(h['X_train'][:,-8:,:],axis=(1,2))==0
    with np.load(GRAY/'splits.npz') as z:
        flagged={k:z[k][flag[z[k]]].tolist() for k in z}
    with np.load(source/'frame_estimates.npz') as z:
        test=z['h5_test_indices']
        ids=test[~flag[test]]
        d=z['h5_distances_m'][ids]
        data={'h5_'+f:[(d,z['h5_'+f+'_m'][ids])] for f in ('native','dino')}
    qs,report={},{}
    for phase,seed,n in [('calibration',202611170518,199),('holdout',202611170519,500)]:
        rng=np.random.default_rng(seed)
        initial=rng.uniform([15,2.5],[16,3],(n,2))
        appearance=rng.integers(12,size=n)
        for actor in ACTORS:
            scores={}
            for front in data:
                key=actor+'__'+front
                r=collect(models,data,front,actor,initial,appearance)
                np.savez_compressed(out/(key+'_'+phase+'.npz'),**r)
                if phase=='calibration':
                    qs[key]={k:float(np.sort(r[k])[189]) for k in ('max_truth_m','max_proxy_m')}
                else:
                    q=qs[key]
                    report[key]=dict(q=q,physical_covered=int((r['max_truth_m']<=q['max_truth_m']).sum()),
                        outcomes=dict(zip(*[v.tolist() for v in np.unique(r['outcomes'],return_counts=True)])),
                        mean_peak_m=float(r['max_truth_m'].mean()))
                    scores[front]=r['max_truth_m']
            if phase=='holdout':
                diff=scores['h5_native']-scores['h5_dino']
                boot=np.random.default_rng(202611170520).choice(diff,(4000,n)).mean(axis=1)
                report[actor+'_paired']=dict(mean_m=float(diff.mean()),bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    favorable=int((diff>0).sum()))
        if phase=='calibration':
            write(out/'frozen_thresholds.json',qs)
    report['flagged_rows']=flagged
    report['frame_mae_m']={k:float(np.abs(v[0][1]-d).mean()) for k,v in data.items()}
    write(out/'report.json',report)
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    run(p.parse_args().source)
