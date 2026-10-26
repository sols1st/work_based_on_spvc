"""Read-only diagnostics of the saved quadrotor plan."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source))
import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from dyn.quad_10d_of_perception import Quad10d_OF
from util.SLS import SLS

path = args.source / 'expe/quad_perception/data/quad10d_perception_0.npz'
plan = np.load(path)
states, inputs = plan['nominal_traj'], plan['nominal_input']
nx, times = states.shape
nu, N = inputs.shape
m = Quad10d_OF()
m.dt = .15
x, u, w = ca.MX.sym('x', nx), ca.MX.sym('u', nu), ca.MX.sym('w', nx)
f = m.ddyn(x, u, w, m.dt)
Afn = ca.Function('A', [x,u], [ca.substitute(ca.jacobian(f,x), w, ca.DM.zeros(nx))])
Bfn = ca.Function('B', [x,u], [ca.substitute(ca.jacobian(f,u), w, ca.DM.zeros(nx))])
A = [np.array(Afn(states[:,k],inputs[:,k])) for k in range(N)]
B = [np.array(Bfn(states[:,k],inputs[:,k])) for k in range(N)]
Zx, Zy = SLS.get_block_downshift_matrix(N+1,nx), SLS.get_block_downshift_matrix(N+1,m.ny)
M = np.eye((N+1)*nx)-Zx@block_diag(*A,np.zeros((nx,nx)))
D = Zx@block_diag(*B,np.zeros((nx,nu)))
H = Zy@block_diag(*([m.C]*N),np.zeros_like(m.C))
xx, ux, xy, uy = [plan[k] for k in ['Phi_xx','Phi_ux','Phi_xy','Phi_uy']]
identities = [M@xx-D@ux-np.eye((N+1)*nx),M@xy-D@uy,
              xx@M-xy@H-np.eye((N+1)*nx),ux@M-uy@H]
defects = np.stack([np.squeeze(m.ddyn(states[:,k],inputs[:,k],h=m.dt))-states[:,k+1]
                    for k in range(N)])
chi = np.vstack([states[:,:-1],inputs])
stage_residual = np.array(m.G)@chi-m.g[:,None]+plan['backoff'].T
terminal_residual = np.array(m.Gf)@states[:,-1]-m.gf+plan['backoff_f']
bounds = np.stack([np.asarray(m.get_disturbance(states[:,k])) for k in range(times)])
obstacles = [(np.array([1.,1.]),.2),(np.array([-1.5,4.]),.2),
             (np.array([.5,3.]),.2),(np.array([-.5,1.]),.2),(np.array([-1.,2.5]),.2)]
half = np.vstack([plan['backoff'][:,:2],plan['backoff_f'][None,:2]])
clearance = min(float(np.min(np.linalg.norm(states[:2].T-c,axis=1)-r-
                             np.linalg.norm(half,axis=1))) for c,r in obstacles)
summary = {
    'plan_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'state_shape': list(states.shape), 'input_shape': list(inputs.shape),
    'all_saved_arrays_finite': all(np.isfinite(plan[k]).all() for k in plan.files),
    'max_nonlinear_nominal_dynamics_defect': float(np.abs(defects).max()),
    'max_tightened_stage_constraint_residual': float(stage_residual.max()),
    'max_tightened_terminal_constraint_residual': float(terminal_residual.max()),
    'minimum_tube_inflated_nominal_XY_obstacle_clearance_m': clearance,
    'SLS_identity_max_abs_residuals_relinearized_at_final_plan': [float(np.abs(a).max()) for a in identities],
    'error_bound_coefficient_shape': list(m.theta_poly.shape),
    'initial_error_bounds': bounds[0].tolist(),
    'nominal_path_min_error_bounds': bounds.min(axis=0).tolist(),
    'nominal_path_max_error_bounds': bounds.max(axis=0).tolist(),
    'nominal_terminal_position': states[:3,-1].tolist(),
    'target_position': [-1.5,2.5,1.4],
    'nominal_terminal_distance_to_target_m': float(np.linalg.norm(states[:3,-1]-[-1.5,2.5,1.4])),
    'note': 'SLS identities use a fresh linearization at the exported trajectory; '
            'responses may refer to the previous SCP local model. This is not a nonlinear safety certificate.'
}
assert summary['all_saved_arrays_finite']
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'plan_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
