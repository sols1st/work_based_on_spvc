"""Trial scaled perception bounds through the original planner in a separate cwd."""
import argparse
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--scene',choices=['parking_lot','quadrotor'],required=True)
parser.add_argument('--factor',type=float,required=True)
args=parser.parse_args()
assert args.factor>0
args.source,args.output=args.source.resolve(),args.output.resolve()
args.output.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(args.source))
os.chdir(args.output)
import numpy as np
import casadi as ca
from scipy.linalg import block_diag
from util.SLS import SLS
if args.scene=='parking_lot':
    from dyn.unicycle_of_perception import Unicycle_OF as Model
    entry=args.source/'expe/parking_lot/main_parking_lot.py'
    filename='unicycle_of_perception_0.npz'
else:
    from dyn.quad_10d_of_perception import Quad10d_OF as Model
    entry=args.source/'expe/quad_perception/main_quad10d_perception.py'
    filename='quad10d_perception_0.npz'
original=Model.get_disturbance
def scaled(self,x,scale=1.):
    value=original(self,x,1.)
    positive=ca.fmax(value,1e-12) if isinstance(x,ca.MX) else np.maximum(value,1e-12)
    return args.factor*scale*positive
Model.get_disturbance=scaled
if args.scene=='parking_lot':
    original_cov=Model.cov_fn
    def scaled_cov(self,x,scale=1.):
        value=original_cov(self,x,1.)
        return args.factor*scale*value.clip(min=1e-12)
    Model.cov_fn=scaled_cov
info={'scene':args.scene,'host':platform.node(),'factor':args.factor,
      'purpose':'Engineering feasibility of bounds from retrospective calibration; not a CP safety claim.',
      'scaling':'q * max(original_bound, 1e-12)', 'original_entry':str(entry)}
started=time.perf_counter()
try:
    runpy.run_path(str(entry),run_name='__main__')
    plan=np.load(args.output/'data'/filename)
    states,inputs=plan['nominal_traj'],plan['nominal_input']
    nx,times=states.shape
    nu,N=inputs.shape
    m=Model()
    m.dt=.15
    if args.scene=='parking_lot':
        m.g=np.concatenate(([.5,1.9,2*np.pi,2.],[np.pi,4.],
                            -np.array([-3.5,-2.,-2*np.pi,-1.]),-np.array([-np.pi,-4.])))
        m.gf=np.concatenate(([.45,.65,2*np.pi,1.],-np.array([0.,.35,-2*np.pi,-1.])))
        obstacles=[(np.array([-3.55,.7]),.4),(np.array([-.6,1.15]),.35),(np.array([-1.2,-1.2]),.4)]
    else:
        obstacles=[(np.array([1.,1.]),.2),(np.array([-1.5,4.]),.2),(np.array([.5,3.]),.2),
                   (np.array([-.5,1.]),.2),(np.array([-1.,2.5]),.2)]
    defects=np.stack([np.squeeze(m.ddyn(states[:,k],inputs[:,k],h=m.dt))-states[:,k+1] for k in range(N)])
    stage=np.array(m.G)@np.vstack([states[:,:-1],inputs])-m.g[:,None]+plan['backoff'].T
    final=np.array(m.Gf)@states[:,-1]-m.gf+plan['backoff_f']
    half=np.vstack([plan['backoff'][:,:2],plan['backoff_f'][None,:2]])
    clearance=min(float(np.min(np.linalg.norm(states[:2].T-c,axis=1)-r-np.linalg.norm(half,axis=1)))
                  for c,r in obstacles)
    x,u,w=ca.MX.sym('x',nx),ca.MX.sym('u',nu),ca.MX.sym('w',nx)
    f=m.ddyn(x,u,w,m.dt)
    af=ca.Function('a',[x,u],[ca.substitute(ca.jacobian(f,x),w,ca.DM.zeros(nx))])
    bf=ca.Function('b',[x,u],[ca.substitute(ca.jacobian(f,u),w,ca.DM.zeros(nx))])
    A=[np.array(af(states[:,k],inputs[:,k])) for k in range(N)]
    B=[np.array(bf(states[:,k],inputs[:,k])) for k in range(N)]
    Zx,Zy=SLS.get_block_downshift_matrix(N+1,nx),SLS.get_block_downshift_matrix(N+1,m.ny)
    M=np.eye((N+1)*nx)-Zx@block_diag(*A,np.zeros((nx,nx)))
    D=Zx@block_diag(*B,np.zeros((nx,nu)))
    H=Zy@block_diag(*([m.C]*N),np.zeros_like(m.C))
    xx,ux,xy,uy=[plan[k] for k in ['Phi_xx','Phi_ux','Phi_xy','Phi_uy']]
    ids=[M@xx-D@ux-np.eye((N+1)*nx),M@xy-D@uy,xx@M-xy@H-np.eye((N+1)*nx),ux@M-uy@H]
    info.update(planner_returned_success=True,all_arrays_finite=all(np.isfinite(plan[k]).all() for k in plan.files),
                state_shape=list(states.shape),max_nonlinear_dynamics_defect=float(np.abs(defects).max()),
                max_tightened_stage_residual=float(stage.max()),max_tightened_terminal_residual=float(final.max()),
                minimum_inflated_obstacle_clearance=clearance,
                sls_residuals_relinearized_at_export=[float(np.abs(a).max()) for a in ids],
                initial_scaled_bounds=np.asarray(m.get_disturbance(states[:,0])).reshape(-1).tolist())
except Exception as exc:
    traceback.print_exc()
    info.update(planner_returned_success=False,error_type=type(exc).__name__,error_message=str(exc))
info['wall_seconds']=time.perf_counter()-started
(args.output/'feasibility.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info,indent=2),flush=True)
