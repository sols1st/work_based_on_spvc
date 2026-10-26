"""Fit a new quadratic quantile predictor from old reference-policy raw traces."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack,vstack
from state_cp_features import features,design,predict

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve()
protocol=json.loads((root/'protocol.json').read_text())
training_task=protocol['training']['source_task']
source=root.parent/training_task/'results/calibration/diagnostic_traces.npz'
d=np.load(source)
states=d['observation_states'].transpose(0,2,1).reshape(-1,d['states'].shape[1])
target=np.linalg.norm(d['residuals'],axis=1).reshape(-1)
f=features(states,protocol['scene'])
model={'scene':protocol['scene'],'feature_mean':f.mean(axis=0).tolist(),
       'feature_scale':np.maximum(f.std(axis=0),.05).tolist(),
       'feature_train_min':f.min(axis=0).tolist(),'feature_train_max':f.max(axis=0).tolist(),
       'target_scale':max(float(np.median(target)),1e-6),'floor':.001,
       'degree':2,'quantile_tau':.9,'l1_regularization':.001,
       'training_traces_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
       'old_envelope_used':False,'training_trajectories':len(d['states'])}
X=design(states,model);n,k=X.shape
# target = X beta + positive residual - negative residual
c=np.r_[np.zeros(k),np.full(n,.9/n),np.full(n,.1/n),np.full(k-1,.001)]
Aeq=hstack((csr_matrix(X),eye(n),-eye(n),csr_matrix((n,k-1))),format='csr')
select=np.eye(k)[1:]
Aub=csr_matrix(np.block([[select,np.zeros((k-1,2*n)),-np.eye(k-1)],
                        [-select,np.zeros((k-1,2*n)),-np.eye(k-1)]]))
t0=time.perf_counter()
sol=linprog(c,A_ub=Aub,b_ub=np.zeros(2*(k-1)),A_eq=Aeq,
            b_eq=target/model['target_scale'],bounds=[(None,None)]*k+[(0,None)]*(2*n+k-1),
            method='highs',options={'time_limit':60})
assert sol.success,sol.message
model.update(coefficients=sol.x[:k].tolist(),fit_wall_seconds=time.perf_counter()-t0,
             fitting_points=n,monomials=k,lp_objective=float(sol.fun))
yp=predict(states,model)
model.update(training_point_coverage=float(np.mean(target<=yp+1e-8)),
             training_predictor_min=float(yp.min()),training_predictor_max=float(yp.max()),
             training_residual_min=float(target.min()),training_residual_max=float(target.max()))
(root/'results/state_predictor.json').write_text(json.dumps(model,indent=2)+'\n')
print(json.dumps(model,indent=2),flush=True)
