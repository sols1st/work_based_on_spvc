"""Compare frozen SPVC and DINO observations under the release's full 24k replay law."""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from Aebs.experiments.native_spvc_camera_cross import (
    GEN, H5, PACKAGE, RELEASE, RGB_GRAY, STATE, Models,
    barrier_status, collect, meters, read, tensor, write,
)
from Aebs.vision_sls.finalize_robust_aebs import inclusion_evidence
from Aebs.vision_sls.run import sha256
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint


@torch.inference_mode()
def run(out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    files = [Path(__file__), Path(__file__).with_name('native_spvc_camera_cross.py'), GEN, STATE, H5, RGB_GRAY,
        PACKAGE/'observations.npz',PACKAGE/'observation.pt',PACKAGE/'policy.pt',PACKAGE/'sbc.pt',
        PACKAGE/'config.json',RELEASE/'development_certificate.json',RELEASE/'development_contract.json',
        Path('Aebs/system/outcomes.py'), Path('Aebs/vision_sls/verify_sbc_cells.py')]
    binding = {str(f):sha256(f) for f in files}
    write(out/'protocol.json',dict(inputs=binding, calibration_n=199, holdout_n=500, rank=190, alpha=.05,
        seeds=[202611170515,202611170516], arms=['rgb_native','rgb_dino'], actor='release robust_actor',
        horizon=400, initial='iid uniform [15,16] x [2.5,3]', appearance='one iid uniform pair among 12 fixed weather/colors per episode',
        renderer='all 24,000 train/val/test images; nearest distance within fixed appearance, same file ties as release',
        metrics='maximum active-frame abs error vs current physical state and vs own nominal observation; all failures retained',
        proof='physical distance tube for frozen release actor and B; check inclusion or search actual witness',
        scope='conditional on frozen deployment bank, containing model-fitting images; no fresh-image generalization'))
    models = Models()
    data = {}
    with np.load(PACKAGE/'observations.npz') as z, np.load(RGB_GRAY) as raw:
        for a,b in [('files','filename'),('distances','distance_m'),('split','split')]:
            if not np.array_equal(raw[a],z[b]):
                raise ValueError('bank identity mismatch')
        groups = sorted(set(zip(z['weather'].tolist(),z['color'].tolist())))
        by_group = dict(zip(groups,range(len(groups))))
        appearance = np.array([by_group[k] for k in zip(z['weather'],z['color'])])
        if len(groups)!=12 or len(appearance)!=24000:
            raise ValueError('unexpected release library')
        native = meters(np.concatenate([models.estimate_native(chunk) for chunk in
            tensor(raw['images']).split(256)]),models.original_scale)
        for name,estimate in [('rgb_native',native),('rgb_dino',meters(z['observation'],models.robust.distance_scale_m))]:
            data[name] = [(z['distance_m'][appearance==i],estimate[appearance==i]) for i in range(len(groups))]
        np.savez_compressed(out/'estimates.npz',files=z['filename'],distance_m=z['distance_m'],native_m=native,
            dino_m=meters(z['observation'],models.robust.distance_scale_m),appearance=appearance,split=z['split'])
    write(out/'groups.json',groups)
    outer = read(RELEASE/'development_contract.json')
    parent = read(RELEASE/'development_certificate.json')
    if parent['source_contract_sha256']!=sha256(RELEASE/'development_contract.json'):
        raise ValueError('parent physical error contract')
    inclusion_evidence(parent,outer,outer,PACKAGE/'policy.pt',PACKAGE/'sbc.pt',PACKAGE/'observation.pt')
    cfg = read(PACKAGE/'config.json')
    barrier,_ = load_sbc_checkpoint(PACKAGE/'sbc.pt',cfg)
    barrier = barrier.double().eval()
    actor64 = copy.deepcopy(models.robust).double()
    thresholds,report = {},{}
    for stage,seed,n in [('calibration',202611170515,199),('holdout',202611170516,500)]:
        rng = np.random.default_rng(seed)
        initial = rng.uniform([15,2.5],[16,3],(n,2))
        appearance = rng.integers(len(groups),size=n)
        for name in data:
            r = collect(models,data,name,'robust_actor',initial,appearance)
            np.savez_compressed(out/(name+'_'+stage+'.npz'),**r)
            if stage=='calibration':
                thresholds[name] = {k:float(np.sort(r[k])[189]) for k in ['max_truth_m','max_proxy_m']}
            else:
                q = thresholds[name]
                counts = dict(zip(*[a.tolist() for a in np.unique(r['outcomes'],return_counts=True)]))
                report[name] = dict(q=q, covered={k:int(np.sum(r[k]<=v)) for k,v in q.items()},
                    episodes=n,outcomes=counts,max_lookup_m=float(r['max_lookup_m'].max()),
                    mean_trajectory_peak_m=float(np.mean(r['max_truth_m'])),
                    barrier=barrier_status(q['max_truth_m'],actor64,barrier,cfg,parent,outer))
        if stage=='calibration':
            write(out/'frozen_thresholds.json',thresholds)
    with np.load(out/'rgb_native_holdout.npz') as old, np.load(out/'rgb_dino_holdout.npz') as new:
        if not np.array_equal(old['initial'],new['initial']) or not np.array_equal(old['appearance'],new['appearance']):
            raise ValueError('pairing')
        diff=old['max_truth_m']-new['max_truth_m']
        boot=np.random.default_rng(202611170517).choice(diff,(4000,len(diff))).mean(axis=1)
        report['paired']=dict(mean_peak_reduction_m=float(diff.mean()),favorable=int((diff>0).sum()),
            bootstrap95=np.quantile(boot,[.025,.975]).tolist())
    if any(sha256(f)!=binding[str(f)] for f in files):
        raise ValueError('inputs changed')
    write(out/'report.json',report)
    print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    run(parser.parse_args().output_dir)
