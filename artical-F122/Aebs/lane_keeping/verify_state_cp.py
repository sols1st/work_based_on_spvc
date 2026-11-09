"""Numerical SBC audit of the actual uncapped state-dependent lane error set."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from . import core
from .state_dependent_cp import scale,write
from .learned_sbc import bounds,grid_boxes,unsafe_boxes,transitions,error_boxes,sha256,load_sbc_checkpoint
from .policy import load_actor
from .verify_learned import region_check
from .certify_neural import split


def radius(states,contract):
    physical=states.detach().cpu().numpy()*[core.E_SCALE,core.THETA_SCALE]
    return states.new_tensor(contract['q_hat']*scale(physical,contract['coefficients']))

def box_radius(lo,hi,contract):
    # Bilinear Bernstein on each box: its maximum is at a vertex.
    vertices=torch.stack([lo,hi,torch.stack([lo[:,0],hi[:,1]],1),torch.stack([hi[:,0],lo[:,1]],1)])
    return radius(vertices,contract).amax(0)+1e-12

@torch.no_grad()
def audit(actor,barrier,contract,kappa,grid=40,depth=16,splits=8):
    lo,hi=grid_boxes([-1,-1],[1,1],grid);unit_errors=error_boxes(1.,splits)
    accepted_area=0.;checked=0;history=[];witness=[]
    for level in range(depth+1):
        pending_l=[];pending_h=[]
        for low,high in zip(lo.split(256),hi.split(256)):
            checked+=len(low);bl,_=bounds(barrier,low,high)
            outside=bl>20;accepted_area+=float((high-low).prod(1)[outside].sum())
            low,high,bl=low[~outside],high[~outside],bl[~outside]
            if not len(low):continue
            r=box_radius(low,high,contract)[:,None];nhi=torch.zeros(len(low),dtype=low.dtype)
            for el,eh in unit_errors:
                nl,nh=transitions(actor,low,high,kappa,r*el,r*eh)
                nhi=torch.maximum(nhi,bounds(barrier,nl,nh)[1])
            good=nhi<=bl;accepted_area+=float((high-low).prod(1)[good].sum())
            low,high=low[~good],high[~good]
            pending_l.append(low);pending_h.append(high)
            if len(low) and level>=2:
                x=(low+high)/2;rad=radius(x,contract);current=barrier(x).flatten()
                for a in [-1.,0.,1.]:
                    for b in [-1.,0.,1.]:
                        err=rad[:,None]*x.new_tensor([a,b]);obs=torch.cat([x+err,x.new_full((len(x),1),kappa/core.KAPPA_SCALE)],1)
                        u=actor(obs).flatten().numpy()*core.PHI_MAX
                        nxt=core.dynamics_batch(x.numpy()*[core.E_SCALE,core.THETA_SCALE],u,np.full(len(x),kappa))/[core.E_SCALE,core.THETA_SCALE]
                        drift=barrier(x.new_tensor(nxt)).flatten()-current
                        bad=(current<=20)&(drift>1e-9)
                        for st,e,d in zip(x[bad].tolist(),err[bad].tolist(),drift[bad].tolist()):witness.append(dict(state=st,error=e,drift=d,kappa=kappa))
        pl=torch.cat(pending_l) if pending_l else torch.empty((0,2),dtype=torch.float64)
        ph=torch.cat(pending_h) if pending_h else pl.clone()
        entry=dict(depth=level,checked=checked,unresolved=len(pl),actual_witnesses=len(witness));history.append(entry);print(json.dumps(dict(kappa=kappa,**entry)),flush=True)
        if not len(pl) or witness or level==depth or len(pl)>100000:
            area_error=abs(accepted_area+float((ph-pl).prod(1).sum())-4)
            return dict(kappa=kappa,passed=not len(pl) and not witness and area_error<1e-8,unresolved=len(pl),actual_witnesses=len(witness),partition_area_error=area_error,history=history,witnesses=witness)
        lo,hi=split(pl,ph)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','contract','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv);args.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    c=json.loads(args.contract.read_text())
    if c['format']!='lane_state_dependent_cp_v2' or c['cap'] is not None:raise ValueError('uncapped repaired contract required')
    if c['envelope_units']!='normalized_observation':raise ValueError('wrong envelope units')
    for name in ['actor.pt','sbc.pt','patch_projection.pt']:
        if c['source_binding'][name]!=sha256(args.source/name):raise ValueError('model identity mismatch: '+name)
    actor=load_actor(args.source/'actor.pt').double().requires_grad_(False)
    barrier,_=load_sbc_checkpoint(args.source/'sbc.pt');barrier=barrier.double().requires_grad_(False)
    lo,hi=grid_boxes([-core.INIT_E/core.E_SCALE,-core.INIT_THETA/core.THETA_SCALE],[core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE],16)
    initial,_=region_check(barrier,lo,hi,True,'affine');lo,hi=unsafe_boxes(16);unsafe,_=region_check(barrier,lo,hi,False,'affine')
    reach=[1+core.SPEED*core.DT/core.E_SCALE*np.sin(core.THETA_SCALE),1+core.SPEED*core.DT/(core.WHEELBASE*core.THETA_SCALE)*np.tan(core.PHI_MAX)+core.SPEED*core.DT*max(abs(k) for k in core.KAPPAS)/core.THETA_SCALE]
    if max(reach)>=1.3:raise ValueError('unsafe audit does not cover first exits')
    checks=[audit(actor,barrier,c,k) for k in core.KAPPAS]
    result=dict(passed=initial['passed'] and unsafe['passed'] and all(x['passed'] for x in checks),initial=initial,unsafe=unsafe,checks=checks,
        contract_sha256=sha256(args.contract),binding=c['source_binding'],
        radius_mode='uncapped state-dependent; bilinear maximum at four box vertices',
        source_sha256={name:sha256(Path(__file__).with_name(name)) for name in ['verify_state_cp.py','state_dependent_cp.py','core.py','learned_sbc.py']},
        scope='Conditional numerical SBC audit only, not an outward-rounded IEEE proof or CP coverage proof')
    write(args.output_dir/'certificate.json',result);print(json.dumps(dict(passed=result['passed'])),flush=True)

if __name__=='__main__':main()
