"""Check the replacement's construction and observation geometry."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import casadi as ca

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from direct_cp_model import install_direct_cp
Model=install_direct_cp(root/'source',.2)
with patch('numpy.load',side_effect=AssertionError('Old envelope must not be read')):
    m=Model()
assert not hasattr(m,'theta_poly') and not hasattr(m,'theta_bias')
assert m.C.shape==(3,4) and np.array_equal(m.F,.2*np.eye(3))
x=ca.MX.sym('x',4)
v=ca.MX.sym('v',3)
y=m.measurement(x,v)
J=ca.Function('J',[x,v],[ca.jacobian(y,v),ca.jacobian(y,x)])
F,C=J(np.zeros(4),np.zeros(3))
assert np.array_equal(np.array(F),.2*np.eye(3))
assert np.array_equal(np.array(C),m.C)
assert np.array_equal(m.cov_fn(np.zeros((5,2))),np.full((5,1),.2))
record={'old_envelope_load_blocked_during_construction':True,
        'old_polynomial_attributes_absent':True,
        'symbolic_observation_jacobians_checked':True,
        'constant_bound_plotting_checked':True,'C':m.C.tolist()}
(root/'results/model_checks.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
