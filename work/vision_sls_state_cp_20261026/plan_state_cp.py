"""Validate and plan with the new state-dependent bound in an isolated output."""
import argparse,json,os,runpy,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import casadi as ca
from state_cp_features import bound
from state_cp_model import install_state_cp

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--convexify',action='store_true')
a=p.parse_args();root=a.root.resolve();source=(root/'source').resolve()
modelpath=root/'results/state_cp_model.json';meta=json.loads(modelpath.read_text())
sys.path.insert(0,str(source))
Model=install_state_cp(source,modelpath)
with patch('numpy.load',side_effect=AssertionError('Old bound load forbidden')),patch('torch.load',side_effect=AssertionError('Old bound load forbidden')):
    m=Model()
assert not hasattr(m,'theta_poly') and not hasattr(m,'theta_bias')
train_task=json.loads((root/'protocol.json').read_text())['training']['source_task']
d=np.load(root.parent/train_task/'results/calibration/diagnostic_traces.npz')
states=d['observation_states'].transpose(0,2,1).reshape(-1,m.nx)[::23]
numeric=bound(states,meta)
symbolic=np.array([float(m.state_cp_bound_fn(x)) for x in states])
assert np.allclose(numeric,symbolic,atol=1e-10,rtol=1e-10)
assert np.ptp(numeric)>1e-5,'Bound collapsed to a constant on training support'
x,v=ca.MX.sym('x',m.nx),ca.MX.sym('v',m.ny)
y=m.measurement(x,v)
jac=ca.Function('measurement_jac',[x,v],[ca.jacobian(y,v),ca.jacobian(y,x)])
for state in states[::max(1,len(states)//5)]:
    F,C=jac(state,np.zeros(m.ny))
    assert np.allclose(F,float(m.state_cp_bound_fn(state))*np.eye(m.ny))
    assert np.allclose(C,m.C)
checks={'old_bound_loading_blocked':True,'numeric_symbolic_bound_equal':True,
        'nonconstant_bound_checked':True,'symbolic_measurement_jacobians_checked':True,
        'training_support_bound_min':float(numeric.min()),'training_support_bound_max':float(numeric.max())}
(root/'results/model_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
out=root/'results/state_cp_plan';out.mkdir(exist_ok=True);os.chdir(out)
entry='expe/parking_lot/main_parking_lot.py' if meta['scene']=='parking' else 'expe/quad_perception/main_quad10d_perception.py'
filename='unicycle_of_perception_0.npz' if meta['scene']=='parking' else 'quad10d_perception_0.npz'
from qp_audit import install_qp_audit
install_qp_audit(root/'results/qp_audit.json',a.convexify)
t=time.perf_counter()
runpy.run_path(str(source/entry),run_name='__main__')
elapsed=time.perf_counter()-t
sys.argv=['check_plan.py','--source',str(source),'--output',str(out),'--plan',str(out/'data'/filename)]
runpy.run_path(str(root/'check_plan.py'),run_name='__main__')
(out/'state_cp_plan.json').write_text(json.dumps({'model':str(modelpath),'wall_seconds':elapsed,
   'old_polynomial_loaded':False,'state_dependent':True},indent=2)+'\n')
print('State-dependent CP planning complete.',flush=True)
