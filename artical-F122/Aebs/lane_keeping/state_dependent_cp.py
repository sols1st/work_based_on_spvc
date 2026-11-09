"""Disjoint-fit trajectory CP; uncapped positive state-dependent scale.
Reused holdout is regression, not newly collected independent confirmation.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack
from .cp import conformal_radius
from .learned_sbc import sha256
from . import core

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def basis(states):
    """Physical state -> signed normalized domain -> [0,1]^2; order 00,01,10,11.
    Clipped extension outside safe domain keeps failed trajectories scoreable.
    """
    x=np.asarray(states,dtype=np.float64)
    if x.shape[-1]!=2 or not np.isfinite(x).all():raise ValueError('finite physical states required')
    t=np.clip((x/[core.E_SCALE,core.THETA_SCALE]+1)/2,0,1)
    a=np.stack([1-t[...,0],t[...,0]],-1);b=np.stack([1-t[...,1],t[...,1]],-1)
    return (a[..., :,None]*b[...,None,:]).reshape(x.shape[:-1]+(4,))

def arrays(data):
    states=np.asarray(data['states'],dtype=np.float64);errors=np.asarray(data['errors'],dtype=np.float64)
    if errors.ndim!=3 or errors.shape[-1]!=2 or min(errors.shape[:2])<1:raise ValueError('complete nonempty [episodes,time,2] errors required')
    if states.shape!=(errors.shape[0],errors.shape[1]+1,2):raise ValueError('initial and all successors required')
    if not np.isfinite(errors).all() or not np.isfinite(states).all():raise ValueError('nonfinite records cannot be discarded')
    return states[:,:-1],np.abs(errors).max(axis=2)

def scale(states,coefficients):
    c=np.asarray(coefficients,dtype=float)
    if c.shape!=(4,) or not np.isfinite(c).all() or np.any(c<=0):raise ValueError('four strictly positive coefficients required')
    return basis(states)@c

def scores(data,coefficients):
    states,residual=arrays(data)
    return (residual/scale(states,coefficients)).max(axis=1)

def fit(data,floor=.005,slack_penalty=20.):
    states,residual=arrays(data);B=basis(states).reshape(-1,4);n=len(B)
    result=linprog(np.r_[B.mean(0),np.full(n,slack_penalty/n)],
        A_ub=hstack([-csr_matrix(B),-eye(n,format='csr')],format='csr'),b_ub=-residual.reshape(-1),
        bounds=[(floor,None)]*4+[(0,None)]*n,method='highs')
    if not result.success:raise RuntimeError(result.message)
    return result.x[:4]

def subset(data,ids):return {k:np.asarray(data[k])[ids] for k in ['states','errors']}

def summary(data,c,q):
    states,_=arrays(data);r=q*scale(states,c);s=scores(data,c)
    return dict(n=len(s),covered=int((s<=q).sum()),fraction=float((s<=q).mean()),
        mean_radius=float(r.mean()),observed_max_radius=float(r.max()),failed_trajectories_removed=0)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv);out=args.output_dir;out.mkdir(parents=True,exist_ok=False);src=args.source
    binding={name:sha256(src/name) for name in ['protocol.json','calibration.npz','holdout.npz','actor.pt','sbc.pt','patch_projection.pt']}
    protocol=dict(format='lane_state_cp_repair_v2',fit_ids=list(range(99)),calibration_ids=list(range(99,199)),alpha=.05,degree=1,floor=.005,slack_penalty=20.,cap=None,
        basis='t=((e/0.85,theta/0.30)+1)/2 clipped to [0,1]; order 00,01,10,11',
        score='max over all time and both normalized error coordinates of abs(error)/b(physical state)',
        holdout_use='Previously inspected 100 trajectories: regression only, not fresh confirmation',
        selection='One prescribed repair; no search over degree, cap, split or holdout coverage',source=str(src),binding=binding,source_sha256=sha256(Path(__file__)))
    write(out/'protocol.json',protocol)
    data=dict(np.load(src/'calibration.npz',allow_pickle=False))
    if len(data['errors'])!=199:raise ValueError('requires registered 199 episodes')
    fit_data=subset(data,protocol['fit_ids']);cal=subset(data,protocol['calibration_ids']);c=fit(fit_data);cal_scores=scores(cal,c)
    q,rank=conformal_radius(cal_scores,.05)
    if not np.isfinite(q):raise ValueError('infinite quantile: no finite certificate')
    # Global analytic bound: nonnegative Bernstein basis sums to one.
    bound=float(q*c.max())
    contract=dict(format='lane_state_dependent_cp_v2',coefficients=c.tolist(),q_hat=q,rank=rank,score_units='dimensionless',envelope_units='normalized_observation',cap=None,
        global_radius_bound=bound,domain='normalized safe-state box [-1,1]^2 with clipped extension outside',basis=protocol['basis'],fit_n=99,calibration=summary(cal,c,q),
        source_binding=binding,protocol_sha256=sha256(out/'protocol.json'),
        guarantee='Split-CP marginal coverage under fixed protocol and exchangeable calibration/new trajectories conditional on disjoint fit data; not fixed-state conditional coverage or fresh empirical confirmation.')
    write(out/'perception_contract.json',contract);contract_hash=sha256(out/'perception_contract.json')
    np.savez_compressed(out/'calibration_scores.npz',scores=cal_scores,ids=np.array(protocol['calibration_ids']))
    # Threshold frozen before reading holdout; no optimization below.
    hold=dict(np.load(src/'holdout.npz',allow_pickle=False));hold_scores=scores(hold,c)
    np.savez_compressed(out/'holdout_scores.npz',scores=hold_scores)
    cq,crank=conformal_radius(np.abs(cal['errors']).max(axis=(1,2)),.05)
    report=dict(contract=contract,holdout_regression=summary(hold,c,q),
        constant_same_calibration=dict(n=100,rank=crank,radius=cq,holdout_covered=int((np.abs(hold['errors']).max(axis=(1,2))<=cq).sum())),
        sbc_verification='pending separate complete audit at global_radius_bound',fresh_holdout=False)
    assert contract_hash==sha256(out/'perception_contract.json')
    assert binding=={name:sha256(src/name) for name in binding}
    write(out/'report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
