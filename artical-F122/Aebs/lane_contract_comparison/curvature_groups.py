"""Secondary split calibration by known curvature magnitude on archived image trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audit import dynamic
from .collect import ARMS,maxima,parameters,scores
from ..lane_keeping import core
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.learned_sbc import load_sbc_checkpoint,sha256
from ..lane_keeping.policy import load_actor
from ..lane_paired_synthesis.evaluate import read,save


def analyze(source, package, out):
    out.mkdir(exist_ok=False,parents=True)
    proto=read(source/'image_protocol.json')
    hashlock=read(source/'all_tubes_frozen.json')
    if hashlock['protocol']!=sha256(source/'image_protocol.json'):
        raise ValueError('image protocol mismatch')
    net=load_actor(package/'actor.pt').double().requires_grad_(False)
    barrier=load_sbc_checkpoint(package/'sbc.pt')[0].double().requires_grad_(False)
    for key,expected in proto['actor_barrier']['fixed'].items():
        assert sha256(package/(key+'.pt'))==expected
    outer=read(source/'five_curve_audit.json')
    assert outer['initial']['passed'] and outer['unsafe']['passed'] and max(outer['first_exit'])<1.3
    rule=dict(groups=[0.,.01,.02],alpha=.05,controls='pool signs within each magnitude; known road curvature; verify both signs',
              extra_rad=[0.,.045,.0675,.0855],source=sha256(Path(__file__)),image_protocol=hashlock['protocol'],
              scope='exploratory subgroup repartition of the existing fixed-policy data; same networks and marginal coverage target')
    save(out/'protocol.json',rule)
    torch.set_num_threads(2)
    fits,counts={},{}
    scale=proto['development']['scale']
    for split in ('calibration','holdout'):
        with np.load(source/'fixed'/(split+'.npz')) as z:
            assert str(z['protocol'])==hashlock['protocol']
            s=scores(maxima(z,net),scale)
            for g in rule['groups']:
                label=str(g)
                mask=np.abs(z['kappas'])==g
                if split=='calibration':
                    fits[label]={}
                    for arm in ARMS:
                        q,rank=conformal_radius(s[arm][mask],rule['alpha'])
                        if not np.isfinite(q): raise ValueError('insufficient group calibration samples')
                        fits[label][arm]=dict(q=q,rank=rank,n=int(mask.sum()),**parameters(arm,q,scale))
                else:
                    counts[label]={arm:dict(covered=int((s[arm][mask]<=fits[label][arm]['q']).sum()),n=int(mask.sum())) for arm in ARMS}
        if split=='calibration':
            save(out/'contracts.json',fits)
    checks={}
    for g in rule['groups']:
        curves=[0.] if not g else [-g,g]
        checks[str(g)]={}
        for arm in ARMS:
            rows={}
            for rad in rule['extra_rad']:
                bycurve={}
                for k in curves:
                    existing=next((m for m,r in rows.items() if r[str(k)]['status'].startswith('witness')),None)
                    if existing is None:
                        bycurve[str(k)]=dynamic(net,barrier,k,fits[str(g)][arm],rad/core.PHI_MAX,proto['audit'])
                    else:
                        bycurve[str(k)]=dict(status='witness_by_inclusion',from_rad=existing)
                rows[rad]=bycurve
            checks[str(g)][arm]=rows
            print(g,arm,{r:[v['status'] for v in by.values()] for r,by in rows.items()},flush=True)
    save(out/'summary.json',dict(fits=fits,holdout=counts,audit=checks,
                                 hashes={n:sha256(source/'fixed'/(n+'.npz')) for n in ['calibration','holdout']},
                                 protocol=sha256(out/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('package',type=Path)
    parser.add_argument('output',type=Path)
    ns=parser.parse_args()
    analyze(ns.source,ns.package,ns.output)
