"""Compare action-error extrema for every admissible initial state under three contracts."""
import argparse
import json
from pathlib import Path
from itertools import product
import torch
from .audit import actor_interval
from .collect import ROOT
from ..lane_keeping import core
from ..lane_keeping.learned_sbc import grid_boxes,sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json


@torch.inference_mode()
def bound(actor,arm,grid):
    # Upper certificate over the full initial rectangle, all error values and curvatures.
    maxlo,maxhi=0.,0.
    ex,tx=core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE
    lowers,uppers=grid_boxes([-ex,-tx],[ex,tx],grid)
    width=lowers.new_tensor(arm['width'])
    witness=None
    for kappa in core.KAPPAS:
        for lo,hi in zip(lowers.split(512),uppers.split(512)):
            floor,ceil=actor_interval(actor,lo-width,hi+width,kappa)
            nomlo,nomhi=actor_interval(actor,lo,hi,kappa)
            absupper=torch.maximum(ceil-nomlo,nomhi-floor).clamp_min(0)
            if arm['action'] is not None:
                absupper=absupper.clamp_max(arm['action'])
            maxhi=max(maxhi,float(absupper.max()))
            # Sample only to witness attainable LOWER bounds. Include a dense radial mesh for the joint arm.
            x=(lo+hi)/2
            curve=x.new_full((len(x),1),kappa/core.KAPPA_SCALE)
            nominal=actor(torch.cat([x,curve],1)).flatten()
            for i,j in product([-1.,0.,1.],repeat=2):
                for scale in ([.125,.25,.375,.5,.625,.75,.875,1.] if arm['action'] is not None else [1.]):
                    eta=width*width.new_tensor([i,j])*scale
                    variant=actor(torch.cat([x+eta,curve],1)).flatten()
                    diff=(variant-nominal).abs()
                    if arm['action'] is not None:
                        diff=torch.where(diff<=arm['action'],diff,0.)
                    value,idx=diff.max(0)
                    if value>maxlo:
                        maxlo=float(value)
                        witness=dict(state=x[int(idx)].tolist(),error=eta.tolist(),kappa=kappa,
                                     action=float(variant[int(idx)]),nominal=float(nominal[int(idx)]))
    if maxhi+1e-10<maxlo:
        raise RuntimeError('action bound smaller than actual example')
    return dict(attained_lower=maxlo,continuous_upper=maxhi,units='normalized steering; times PHI_MAX in radians',
                witness=witness,lower_rad=maxlo*core.PHI_MAX,upper_rad=maxhi*core.PHI_MAX)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('root',type=Path)
    cli.add_argument('--register',action='store_true')
    a=cli.parse_args()
    token=dict(source=sha256(Path(__file__)),audit=sha256(Path(__file__).with_name('audit.py')),
               protocol=sha256(a.root/'protocol.json'),grid=60,
               scope='all original initial states x, three curvatures, allowed perception errors; command error only')
    path=a.root/'action_range_protocol.json'
    if a.register:
        if path.exists() or (a.root/'contracts.json').exists():
            raise FileExistsError('must supplement before new calibration threshold exists')
        write_json(path,token)
        return
    if json.loads(path.read_text())!=token:
        raise ValueError('supplemental diagnostic protocol changed')
    contracts=json.loads((a.root/'contracts.json').read_text())
    if contracts['protocol']!=token['protocol']:
        raise ValueError('wrong collection protocol')
    torch.set_num_threads(2)
    actor=load_actor(ROOT/'package/actor.pt').double().requires_grad_(False)
    result={name:bound(actor,arm,token['grid']) for name,arm in contracts['arms'].items()}
    write_json(a.root/'action_range.json',dict(arms=result,supplement=sha256(path),
               contracts=sha256(a.root/'contracts.json')))


if __name__=='__main__':
    main()
