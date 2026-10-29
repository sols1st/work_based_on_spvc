"""Adaptive continuous-state numerical interval checks for a nonnegative SBC.

Full live domain is covered; positive actual witnesses and unresolved boxes
are failures. Conditional CP-tube support is not inferred from marginal CP.
Bounds are computed in float64; this is not a validated floating-point proof.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .sbc_cp import NonnegativeSBC,monotone_box_bounds,load_sbc_checkpoint
from .deploy import VisionSLSController
from .conformal import envelope
from .cp_ncbf import sequential_bounds
from .audit_sbc_cp import region_grid
from .run import sha256


@torch.no_grad()
def actor_box(policy,low,high,radius,subdivisions=32):
    # Union of observation subintervals encloses all possible d+error.
    left=low[:,0]-radius;right=high[:,0]+radius
    edges=left[:,None]+(right-left)[:,None]*torch.linspace(0,1,subdivisions+1,dtype=low.dtype)
    lo=torch.stack([edges[:,:-1]/policy.distance_scale_m,low[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
    hi=torch.stack([edges[:,1:]/policy.distance_scale_m,high[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
    actor=nn.Sequential(*policy.controller_net.mlp_extractor,policy.controller_net.action_net)
    al,au=sequential_bounds(actor,lo,hi)
    return al.reshape(len(low),-1).amin(1).clamp(-3,3),au.reshape(len(low),-1).amax(1).clamp(-3,3)


@torch.no_grad()
def state_boxes(policy,barrier,scale,contract,low,high,observations=None):
    div=low.new_tensor([scale,1.]);mid=(low+high)/2
    coeff=np.array(contract['coefficients']);q=contract['calibration']['q_hat']
    lip=q*(len(coeff)-1)/11*np.max(np.abs(np.diff(coeff)))
    radius=torch.tensor(envelope(mid[:,0],coeff,q),dtype=low.dtype)+lip*(high[:,0]-low[:,0])/2
    if observations is None:
        al,au=actor_box(policy,low,high,radius)
    else:
        actor=nn.Sequential(*policy.controller_net.mlp_extractor,policy.controller_net.action_net)
        al,au=sequential_bounds(actor,torch.stack([observations,low[:,1]],1),torch.stack([observations,high[:,1]],1))
        al=al.flatten().clamp(-3,3);au=au.flatten().clamp(-3,3)
    nl=torch.stack([low[:,0]-.05*high[:,1],(low[:,1]-.05*au).clamp(0,3)],1)
    nu=torch.stack([high[:,0]-.05*low[:,1],(high[:,1]-.05*al).clamp(0,3)],1)
    bl,bu=monotone_box_bounds(barrier,low/div,high/div)
    # Observation uncertainty leaves a nonzero successor-speed width even for
    # arbitrarily small current-state cells. Refine that width explicitly.
    fractions=torch.linspace(0,1,17,dtype=low.dtype)
    ve=nl[:,1,None]+(nu[:,1]-nl[:,1])[:,None]*fractions
    sublo=torch.stack([nl[:,0,None].expand(-1,16),ve[:,:-1]],-1).reshape(-1,2)
    subhi=torch.stack([nu[:,0,None].expand(-1,16),ve[:,1:]],-1).reshape(-1,2)
    _,upper=monotone_box_bounds(barrier,sublo/div,subhi/div)
    nhi=upper.reshape(len(low),16).amax(1)
    return bl,bu,nhi


@torch.no_grad()
def actual_witnesses(policy,barrier,scale,contract,states,observations=None):
    div=states.new_tensor([scale,1.]);b=barrier(states/div).flatten()
    r=torch.tensor(envelope(states[:,0],contract['coefficients'],contract['calibration']['q_hat']),dtype=states.dtype)
    observed=(states[:,0,None]+r[:,None]*torch.linspace(-1,1,17,dtype=states.dtype))/policy.distance_scale_m if observations is None else observations[:,None]
    a=policy.forward_observation(observed.reshape(-1,1),states[:,1,None].expand_as(observed).reshape(-1,1)).reshape_as(observed)
    d=(states[:,0]-.05*states[:,1])[:,None].expand_as(observed)
    v=(states[:,1,None]-.05*a).clamp(0,3)
    nxt=torch.stack([d,v],-1)
    delta=barrier(nxt.reshape(-1,2)/div).reshape_as(observed)-b[:,None]
    largest,index=delta.max(1)
    bad=(b<20)&(largest>1e-9)
    return bad,largest,observed[torch.arange(len(states)),index]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--barrier-dir',type=Path,required=True)
    p.add_argument('--perception-contract',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-depth',type=int,default=14)
    p.add_argument('--max-cells',type=int,default=300000)
    p.add_argument('--observation-model',choices=['tube','replay'],default='tube')
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('choose a new output directory')
    args.output_dir.mkdir(parents=True);out=args.output_dir
    torch.set_num_threads(1)
    policy=VisionSLSController(args.artifact_dir).policy.double()
    cfg=json.loads((args.barrier_dir/'config.json').read_text())
    barrier,metadata=load_sbc_checkpoint(args.barrier_dir/'sbc.pt',cfg)
    barrier=barrier.double().eval().requires_grad_(False)
    if metadata.get('policy_sha256') and metadata['policy_sha256']!=sha256(args.artifact_dir/'policy.pt'):
        raise ValueError('SBC was verified against a different policy')
    scale=cfg['barrier_distance_scale_m'];contract=json.loads(args.perception_contract.read_text())
    if contract['source_policy_sha256']!=sha256(args.artifact_dir/'policy.pt'):raise ValueError('actor identity mismatch')
    div=torch.tensor([scale,1.],dtype=torch.float64)
    regions={}
    for name,lo,hi in [('initial',[15,2.5],[16,3]),('unsafe',[5,.5],[6,3])]:
        l,u=region_grid(lo,hi,128)
        lb,ub=monotone_box_bounds(barrier,l.double()/div,u.double()/div)
        regions[name]=dict(lower=float(lb.min()),upper=float(ub.max()))
    observations=None
    if args.observation_model=='replay':
        library=np.load(args.artifact_dir/'observations.npz')
        # np.unique returns the first original index for equal-distance images,
        # matching deterministic argmin tie-breaking in the real replay code.
        distances,first=np.unique(library['distance_m'],return_index=True)
        edges=np.r_[5.,(distances[:-1].astype(float)+distances[1:].astype(float))/2,16.]
        speed_edges=np.linspace(1e-6,3.,65);lower=[];upper=[];observed=[]
        for i,index in enumerate(first):
            left=max(6.,edges[i]);right=min(16.,edges[i+1])
            if left>=right:continue
            for a,b in zip(speed_edges[:-1],speed_edges[1:]):
                lower.append([left,a]);upper.append([right,b]);observed.append(library['observation'][index])
        low=torch.tensor(lower,dtype=torch.float64);high=torch.tensor(upper,dtype=torch.float64)
        observations=torch.tensor(observed,dtype=torch.float64)
    else:
        low,high=region_grid([6,1e-6],[16,3],64);low=low.double();high=high.double()
    checked=passed=excluded=unresolved=0;witnesses=[];history=[];start=time.monotonic()
    for depth in range(args.max_depth+1):
        pending_l=[];pending_u=[];pending_observations=[]
        for begin in range(0,len(low),256):
            l=low[begin:begin+256];u=high[begin:begin+256]
            obs=None if observations is None else observations[begin:begin+256]
            bl,bu,nhi=state_boxes(policy,barrier,scale,contract,l,u,obs)
            inactive=bl>=20
            good=(nhi<=bl)&~inactive
            remaining=~(inactive|good)
            checked+=len(l);passed+=int(good.sum());excluded+=int(inactive.sum())
            if remaining.any():
                rl=l[remaining];ru=u[remaining];center=(rl+ru)/2
                actual_obs=None if obs is None else obs[remaining]
                bad,delta,witness_obs=actual_witnesses(policy,barrier,scale,contract,center,actual_obs)
                for s,value,y in zip(center[bad].tolist(),delta[bad].tolist(),witness_obs[bad].tolist()):
                    witnesses.append(dict(state=s,residual=value,observation=y))
                pending_l.append(rl[~bad]);pending_u.append(ru[~bad])
                if actual_obs is not None:pending_observations.append(actual_obs[~bad])
        pending_l=torch.cat(pending_l) if pending_l else torch.empty((0,2),dtype=torch.float64)
        pending_u=torch.cat(pending_u) if pending_u else torch.empty((0,2),dtype=torch.float64)
        history.append(dict(depth=depth,checked=checked,passed=passed,threshold_excluded=excluded,
                            actual_witnesses=len(witnesses),unresolved=len(pending_l),seconds=time.monotonic()-start))
        print(json.dumps(history[-1]),flush=True)
        if not len(pending_l):break
        if depth==args.max_depth or checked+2*len(pending_l)>args.max_cells:
            unresolved=len(pending_l);break
        normalized_width=(pending_u-pending_l)/torch.tensor([10.,3.],dtype=torch.float64)
        coordinate=normalized_width.argmax(1);row=torch.arange(len(pending_l));middle=(pending_l+pending_u)/2
        left_u=pending_u.clone();left_u[row,coordinate]=middle[row,coordinate]
        right_l=pending_l.clone();right_l[row,coordinate]=middle[row,coordinate]
        low=torch.cat([pending_l,right_l]);high=torch.cat([left_u,pending_u])
        if observations is not None:
            parent_obs=torch.cat(pending_observations);observations=torch.cat([parent_obs,parent_obs])
    region_pass=regions['initial']['upper']<=1 and regions['unsafe']['lower']>=20
    result=dict(regions=regions,region_pass=region_pass,checked_cells=checked,proved_cells=passed,
                threshold_excluded_cells=excluded,actual_witnesses=len(witnesses),unresolved_cells=unresolved,
                continuous_state_numerical_interval_pass=region_pass and not witnesses and not unresolved,
                guarantee_scope=('stopped AEBS, all live d in (6,16], v in (1e-6,3], conditional on observation-tube support' if args.observation_model=='tube' else
                                 'stopped deterministic nearest-image replay over the frozen 400-image library; all continuous initial states, no new-image claim'),
                observation_model=args.observation_model,
                expectation_support_assumption_established=args.observation_model=='replay',full_image_certificate=False,
                floating_point_formal_proof=False,arithmetic='float64 mathematical-network interval model; no outward rounding',
                source_policy_sha256=sha256(args.artifact_dir/'policy.pt'),source_sbc_sha256=sha256(args.barrier_dir/'sbc.pt'),
                history=history,witnesses=witnesses)
    (out/'report.json').write_text(json.dumps(result,indent=2))
    np.savez(out/'counterexamples.npz',states=np.array([x['state'] for x in witnesses]).reshape(-1,2),
             residual=np.array([x['residual'] for x in witnesses]),unresolved_low=pending_l.numpy(),unresolved_high=pending_u.numpy())


if __name__=='__main__':main()
