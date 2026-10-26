"""Independently inspect the saved author plan without altering its solution."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(args.source))
import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from dyn.unicycle_of_perception import Unicycle_OF
from util.SLS import SLS

path = args.source / 'expe/parking_lot/data/unicycle_of_perception_0.npz'
plan = np.load(path)
states, inputs = plan['nominal_traj'], plan['nominal_input']
nx, times = states.shape
nu, N = inputs.shape
m = Unicycle_OF()
m.dt = .15
m.g = np.concatenate(([.5,1.9,2*np.pi,2.], [np.pi,4.],
                      -np.array([-3.5,-2.,-2*np.pi,-1.]), -np.array([-np.pi,-4.])))
m.gf = np.concatenate(([.45,.65,2*np.pi,1.], -np.array([0.,.35,-2*np.pi,-1.])))
x, u, w = ca.MX.sym('x',nx), ca.MX.sym('u',nu), ca.MX.sym('w',nx)
f = m.ddyn(x,u,w,m.dt)
Afn = ca.Function('A',[x,u],[ca.substitute(ca.jacobian(f,x),w,ca.DM.zeros(nx))])
Bfn = ca.Function('B',[x,u],[ca.substitute(ca.jacobian(f,u),w,ca.DM.zeros(nx))])
A = [np.array(Afn(states[:,k],inputs[:,k])) for k in range(N)]
B = [np.array(Bfn(states[:,k],inputs[:,k])) for k in range(N)]
Zx, Zy = SLS.get_block_downshift_matrix(N+1,nx), SLS.get_block_downshift_matrix(N+1,m.ny)
M = np.eye((N+1)*nx) - Zx @ block_diag(*A, np.zeros((nx,nx)))
D = Zx @ block_diag(*B,np.zeros((nx,nu)))
H = Zy @ block_diag(*([m.C]*N),np.zeros_like(m.C))
xx,ux,xy,uy = [plan[k] for k in ['Phi_xx','Phi_ux','Phi_xy','Phi_uy']]
identity_residuals = [M@xx-D@ux-np.eye((N+1)*nx), M@xy-D@uy,
                      xx@M-xy@H-np.eye((N+1)*nx), ux@M-uy@H]
defects = np.stack([np.squeeze(m.ddyn(states[:,k],inputs[:,k],h=.15))-states[:,k+1]
                    for k in range(N)])
chi = np.vstack([states[:,:-1],inputs])
stage_residual = np.array(m.G) @ chi - m.g[:,None] + plan['backoff'].T
terminal_residual = np.array(m.Gf) @ states[:,-1]-m.gf+plan['backoff_f']
obstacles = [(np.array([-3.55,.7]),.4),(np.array([-.6,1.15]),.35),
             (np.array([-1.2,-1.2]),.4)]
half = np.concatenate([plan['backoff'][:,:2],plan['backoff_f'][None,:2]])
inflated_clearance = min(float(np.min(np.linalg.norm(states[:2].T-c,axis=1)-r-
                                               np.linalg.norm(half,axis=1))) for c,r in obstacles)
summary = {
    'plan_file': str(path), 'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'state_shape':list(states.shape),'input_shape':list(inputs.shape),
    'max_nonlinear_nominal_dynamics_defect':float(np.abs(defects).max()),
    'max_tightened_stage_constraint_residual':float(stage_residual.max()),
    'max_tightened_terminal_constraint_residual':float(terminal_residual.max()),
    'minimum_tube_inflated_nominal_obstacle_clearance_m':inflated_clearance,
    'SLS_identity_max_abs_residuals_relinearized_at_final_plan':[
        float(np.abs(a).max()) for a in identity_residuals],
    'all_saved_arrays_finite':all(np.isfinite(plan[k]).all() for k in plan.files),
    'note':'The original solver exports responses from its final local model. '
           'These identities are checked against a fresh linearization at the exported trajectory; '
           'they may differ due to the final SCP update. This is a diagnostic, not a nonlinear certificate.'
}
assert summary['all_saved_arrays_finite']
(args.output / 'plan_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
