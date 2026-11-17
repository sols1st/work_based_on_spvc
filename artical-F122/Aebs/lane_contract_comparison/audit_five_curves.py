"""Verify the frozen lane barrier and a one-control-period heading tolerance."""
import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .audit import actor_interval, dynamic
from ..lane_keeping import core
from ..lane_keeping.learned_sbc import grid_boxes, load_sbc_checkpoint, sha256, unsafe_boxes
from ..lane_keeping.policy import load_actor
from ..lane_keeping.verify_learned import region_check
from ..lane_paired_synthesis.evaluate import ARMS, KAPPAS, PACKAGE, read, save


@torch.inference_mode()
def heading_range(actor, contract, degrees):
    """Continuous upper and feasible lower for |theta_next(image)-theta_next(proxy)|."""
    low=core.INITIAL.low / [core.E_SCALE,core.THETA_SCALE]
    high=core.INITIAL.high / [core.E_SCALE,core.THETA_SCALE]
    lo,hi=grid_boxes(low,high,56)
    width=lo.new_tensor(contract['width'])
    factor=core.SPEED*core.DT/core.WHEELBASE
    angle=core.PHI_MAX
    derivative=factor / math.cos(angle)**2
    upper,lower=0.,0.
    witness=None
    cap=contract['action']
    for k in KAPPAS:
        for l,h in zip(lo.split(384),hi.split(384)):
            nl,nh=actor_interval(actor,l,h,k)
            ol,oh=actor_interval(actor,l-width,h+width,k)
            # Same state and curvature cancel all other terms in the heading dynamics.
            outer=factor*torch.maximum((oh*angle).tan()-(nl*angle).tan(),
                                       (nh*angle).tan()-(ol*angle).tan())
            if cap is not None:
                outer=outer.clamp_max(derivative*cap*angle)
            upper=max(upper,float(outer.max())+1e-12)
            x=(l+h)*.5
            curve=l.new_full((len(x),1),k/core.KAPPA_SCALE)
            nominal=actor(torch.cat([x,curve],dim=1)).flatten()
            for direction in product([-1.,0.,1.],repeat=2):
                for scale in ([.25,.5,.75,1.] if cap is not None else [1.]):
                    eta=width*l.new_tensor(direction)*scale
                    obs=actor(torch.cat([x+eta,curve],dim=1)).flatten()
                    diff=factor*((obs*angle).tan()-(nominal*angle).tan()).abs()
                    if cap is not None:
                        diff=torch.where((obs-nominal).abs()<=cap,diff,0.)
                    val,i=diff.max(dim=0)
                    if val>lower:
                        lower=float(val)
                        witness=dict(state_normalized=x[i].tolist(),error=eta.tolist(),kappa=k,
                                     nominal_angle=float(nominal[i]*angle),image_angle=float(obs[i]*angle))
    if lower>upper: raise ValueError('heading range missed witness')
    threshold=math.radians(degrees)
    return dict(feasible_lower_deg=math.degrees(lower),continuous_upper_deg=math.degrees(upper),
                witness=witness,status='pass' if upper<=threshold else ('witness' if lower>threshold else 'unresolved'),
                certified_additional_steering_rad=max(0.,(threshold-upper)/derivative))


def audit(root, register=False):
    protocol=read(root/'image_protocol.json')
    settings=protocol['audit']
    spec=dict(source=sha256(Path(__file__)),base=sha256(Path(__file__).with_name('audit.py')),
              image_protocol=sha256(root/'image_protocol.json'),heading_initial_cells_per_axis=56,
              numerical='float64 interval enclosures, no directed rounding')
    if register:
        if (root/'all_tubes_frozen.json').exists():
            raise ValueError('register before thresholds are frozen')
        save(root/'audit_protocol.json',spec)
        return
    if spec!=read(root/'audit_protocol.json'):
        raise ValueError('wrong code or protocol')
    lock=read(root/'all_tubes_frozen.json')['contracts']
    for name, expected in protocol['actor_barrier']['fixed'].items():
        assert sha256(PACKAGE/(name+'.pt'))==expected
    torch.set_num_threads(2)
    actor=load_actor(PACKAGE/'actor.pt').double().requires_grad_(False)
    B=load_sbc_checkpoint(PACKAGE/'sbc.pt')[0].double().requires_grad_(False)
    low=core.INITIAL.low/[core.E_SCALE,core.THETA_SCALE]
    high=core.INITIAL.high/[core.E_SCALE,core.THETA_SCALE]
    init=region_check(B,*grid_boxes(low,high,16),True,'affine')[0]
    unsafe=region_check(B,*unsafe_boxes(16),False,'affine')[0]
    excursion=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),
               1+core.SPEED*core.DT/core.THETA_SCALE*(np.tan(core.PHI_MAX)/core.WHEELBASE+max(map(abs,KAPPAS)))]
    assert init['passed'] and unsafe['passed'] and max(excursion)<1.3
    reports={}
    for arm in ARMS:
        path=root/arm/'final_contract.json'
        if sha256(path)!=lock[arm]:
            raise ValueError('frozen contract changed')
        fit=read(path)
        heading=heading_range(actor,fit,settings['heading_requirement_deg'])
        runs={}
        for rad in settings['extra_rad']:
            by_curve={}
            for k in KAPPAS:
                prior=next((v for v,r in runs.items() if r[str(k)]['status'] in ('witness','witness_by_inclusion')),None)
                if prior is None:
                    value=dynamic(actor,B,k,fit,rad/core.PHI_MAX,settings)
                else:
                    value=dict(status='witness_by_inclusion',from_rad=prior)
                by_curve[str(k)]=value
            runs[rad]=by_curve
            print(arm,rad,[v['status'] for v in by_curve.values()],flush=True)
        reports[arm]=dict(heading=heading,barrier=runs)
        save(root/arm/'five_curve_audit.json',reports[arm])
    save(root/'five_curve_audit.json',dict(protocol=spec,image_thresholds=lock,initial=init,
                                            unsafe=unsafe,first_exit=excursion,arms=reports))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('--register',action='store_true')
    args=cli.parse_args()
    audit(args.root,args.register)
