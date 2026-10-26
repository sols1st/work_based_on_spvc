"""Fit exactly the document's normalized convex objective on infinity residuals."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import cvxpy as cp
from doc_cp_features import features,powers,design

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();protocol=json.loads((root/'protocol.json').read_text())
train=protocol['training'];path=root.parent/train['source_task']/'results/calibration/diagnostic_traces.npz'
d=np.load(path);nx=d['states'].shape[1]
states=d['observation_states'].transpose(0,2,1).reshape(-1,nx)
r=np.max(np.abs(d['residuals']),axis=1).reshape(-1)
f=features(states,protocol['scene']);n=len(r)
meta={'scene':protocol['scene'],'ny':3 if protocol['scene']=='parking' else 5,
 'epsilon':protocol['bound']['epsilon'],'degree':train['degree'],
 'feature_mean':f.mean(axis=0).tolist(),'feature_scale':np.maximum(f.std(axis=0),.05).tolist(),
 'feature_train_min':f.min(axis=0).tolist(),'feature_train_max':f.max(axis=0).tolist(),
 'powers':powers(f.shape[1],train['degree']),'gamma':train['gamma'],'mu':train['mu'],
 'training_trajectories':len(d['states']),'training_points':n,
 'training_traces_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'old_envelope_loaded':False,'norm':'infinity','cp_type':'multiplicative trajectory CP'}
M=design(states,meta);beta=cp.Variable(M.shape[1]);xi=cp.Variable(n,nonneg=True);b=M@beta
problem=cp.Problem(cp.Minimize(cp.sum(b)/n+meta['gamma']*cp.sum_squares(beta)+meta['mu']*cp.sum(xi)/n),[b+xi>=r,b>=0])
t=time.perf_counter();problem.solve(solver=cp.OSQP,eps_abs=1e-8,eps_rel=1e-8,max_iter=200000,polishing=True)
assert problem.status in ['optimal','optimal_inaccurate'],problem.status
coeff=np.asarray(beta.value).reshape(-1);slack=np.asarray(xi.value);pred=M@coeff
violation=float(np.max(r-pred-slack));nonnegative_violation=float(-pred.min())
assert max(violation,nonnegative_violation,-float(slack.min()))<1e-5
meta.update(coefficients=coeff.tolist(),monomials=M.shape[1],solver_status=problem.status,
 solver='CVXPY OSQP',fit_wall_seconds=time.perf_counter()-t,objective_value=float(problem.value),
 maximum_training_constraint_violation=violation,training_envelope_min=float(pred.min()),
 training_envelope_max=float(pred.max()),training_raw_infinity_residual_max=float(r.max()),
 training_point_coverage=float(np.mean(r<=pred+1e-8)),training_slack_mean=float(slack.mean()),
 training_slack_max=float(slack.max()))
(root/'results/doc_envelope.json').write_text(json.dumps(meta,indent=2)+'\n')
np.savez_compressed(root/'results/training_fit.npz',residual_infinity=r,envelope=pred,slack=slack)
print(json.dumps(meta,indent=2),flush=True)
