"""Reapply the same lane verifier to fixed-policy controls and curvature strata."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .train import BASE,ARMS,deadline,read
from .audit import summarize
from ..lane_contract_comparison.audit import dynamic
from ..lane_contract_comparison.collect import maxima,scores,parameters
from ..lane_keeping import core
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.policy import load_actor
from ..lane_keeping.learned_sbc import grid_boxes,load_sbc_checkpoint,sha256,unsafe_boxes
from ..lane_keeping.robust_trajectory_cp import write_json
from ..lane_keeping.verify_learned import region_check


def audit(actor,barrier,contract,config,curves):
    initialized=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],
                           [core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],16)
    init=region_check(barrier,*initialized,True,'affine')[0]
    hazard=region_check(barrier,*unsafe_boxes(16),False,'affine')[0]
    x_bound=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),
             1+core.SPEED*core.DT/core.THETA_SCALE*(np.tan(core.PHI_MAX)/core.WHEELBASE+max(abs(k) for k in curves))]
    rows=[]
    for margin in config['margins']:
        curv={}
        start=time.monotonic()
        for k in curves:
            remaining=config['seconds_per_margin_per_arm']-(time.monotonic()-start)
            if remaining<=0:
                curv[k]=dict(status='timeout')
                continue
            try:
                with deadline(remaining):
                    curv[k]=dynamic(actor,barrier,k,contract,margin,config)
            except TimeoutError:
                curv[k]=dict(status='timeout')
        status=summarize(curv.values()) if init['passed'] and hazard['passed'] and max(x_bound)<1.3 else 'separation_unproved'
        rows.append(dict(normalized_margin=margin,steering_rad=margin*core.PHI_MAX,
                         status=status,curvatures=curv,seconds=time.monotonic()-start))
    return dict(initial=init,unsafe=hazard,first_exit=x_bound,rows=rows)


def run(root,model,arm):
    torch.set_num_threads(1)
    registered=read(root/'protocol.json')
    config=registered['final_audit']
    folder=root/model
    target=folder/arm if model.startswith('fixed') else folder
    output=target/'supplemental_audit.json'
    if output.exists(): raise FileExistsError(output)
    fixed=model.startswith('fixed')
    package=BASE if fixed else folder
    cp=read(target/'final_contract.json')
    for n,f in [('actor','actor_sha256'),('sbc','sbc_sha256')]:
        assert sha256(package/(n+'.pt'))==cp[f]
    assert sha256(folder/'calibration.npz')==cp['calibration']
    a=load_actor(package/'actor.pt').double().requires_grad_(False)
    b=load_sbc_checkpoint(package/'sbc.pt')[0].double().requires_grad_(False)
    result=dict(source=sha256(Path(__file__)),solver=sha256(Path(__file__).with_name('audit.py')),
                protocol=sha256(root/'protocol.json'),contract=sha256(target/'final_contract.json'),
                actor=cp['actor_sha256'],barrier=cp['sbc_sha256'])
    if fixed:
        # The compensated policy receives z-b(k); in coordinates relative to g(x),
        # the tested actor, dynamics and bounded-error proof use the identical map.
        result['all']=audit(a,b,cp,config,[0.,-.01,.01,-.02,.02])
    else:
        raw=dict(np.load(folder/'calibration.npz'))
        school=registered['development_scale']
        values=scores(maxima(raw,a),school)[arm]
        result['strata']={}
        for name,curves in [('standard',[0.,-.01,.01]),('extended_only',[-.02,.02])]:
            group=np.isin(raw['kappas'],curves)
            q,rank=conformal_radius(values[group],registered['alpha'])
            fit=dict(q=q,rank=rank,n=int(group.sum()),**parameters(arm,q,school))
            audit_result=audit(a,b,fit,config,curves)
            result['strata'][name]=dict(contract=fit,audit=audit_result)
    write_json(output,result)
    print('completed',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('model')
    p.add_argument('arm',choices=ARMS)
    args=p.parse_args()
    run(args.root,args.model,args.arm)
