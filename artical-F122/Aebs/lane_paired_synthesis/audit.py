"""Continuous-domain common-budget audit of a frozen resynthesized controller and B."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..lane_contract_comparison.audit import dynamic
from ..lane_keeping import core
from ..lane_keeping.learned_sbc import grid_boxes,load_sbc_checkpoint,sha256,unsafe_boxes
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json
from ..lane_keeping.verify_learned import region_check
from .train import ARMS,ROOT,deadline,read


def summarize(curves):
    statuses={row['status'] for row in curves}
    if 'witness' in statuses:return 'witness'
    if statuses=={'pass'}:return 'pass'
    if 'timeout' in statuses:return 'timeout'
    return 'unresolved'


def run(root,name):
    torch.set_num_threads(1)
    out=root/name
    path=out/'audit.json'
    if path.exists():
        raise FileExistsError(path)
    protocol=read(root/'protocol.json')
    if protocol['inputs_sha256'][str(Path(__file__))]!=sha256(Path(__file__)):
        raise ValueError('audit source changed')
    frozen=read(out/'frozen.json')
    if (sha256(out/'actor.pt')!=frozen['actor_sha256'] or
        sha256(out/'sbc.pt')!=frozen['sbc_sha256'] or sha256(root/'protocol.json')!=frozen['protocol']):
        raise ValueError('weights or preregistration changed')
    cp=read(out/'final_contract.json')
    if not (np.isfinite(cp['width']).all() and min(cp['width'])>=0):
        raise ValueError('invalid calibrated error width')
    actor=load_actor(out/'actor.pt').double().requires_grad_(False)
    barrier=load_sbc_checkpoint(out/'sbc.pt')[0].double().requires_grad_(False)
    lo,hi=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],
                     [core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],16)
    init,_=region_check(barrier,lo,hi,True,'affine')
    unsafe,_=region_check(barrier,*unsafe_boxes(16),False,'affine')
    curvatures=protocol['kappas']
    exit_limit=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),
                1+core.SPEED*core.DT/core.THETA_SCALE*(np.tan(core.PHI_MAX)/core.WHEELBASE+max(np.abs(curvatures)))]
    config=protocol['final_audit']
    report=dict(arm=name,protocol=sha256(root/'protocol.json'),frozen=sha256(out/'frozen.json'),
                final_contract=sha256(out/'final_contract.json'),init=init,unsafe=unsafe,first_exit_bound=exit_limit,
                numerical='float64 padded interval audit without directed rounding',rows=[])
    # Same wall-time budget even if the baseline curvatures finish earlier. Record each curvature.
    order=[0.,-.01,.01,-.02,.02]
    assert sorted(order)==sorted(curvatures)
    for margin in config['margins']:
        start=time.monotonic()
        rows={}
        for k in order:
            remaining=config['seconds_per_margin_per_arm']-(time.monotonic()-start)
            if remaining<=0:
                rows[k]=dict(status='timeout')
                continue
            try:
                with deadline(remaining):
                    rows[k]=dynamic(actor,barrier,k,cp,margin,config)
            except TimeoutError:
                rows[k]=dict(status='timeout')
        base=summarize([rows[k] for k in core.KAPPAS])
        extended=summarize(rows.values())
        separation=bool(init['passed'] and unsafe['passed'] and max(exit_limit)<1.3)
        if not separation:
            base=extended='separation_unproved'
        report['rows'].append(dict(normalized_margin=margin,steering_rad=margin*core.PHI_MAX,
                                  original_curvatures=base,five_curvatures=extended,curvatures=rows,
                                  seconds=time.monotonic()-start))
        print(name,margin,base,extended,flush=True)
    write_json(path,report)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('arm',choices=ARMS)
    cli.add_argument('--root',type=Path,default=ROOT)
    args=cli.parse_args()
    run(args.root,args.arm)
