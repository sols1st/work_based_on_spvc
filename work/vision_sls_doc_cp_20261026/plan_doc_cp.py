"""Run original planner with a CP box's ball enclosure; audit without convexifying."""
import argparse,json,os,runpy,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
import casadi as ca
from doc_cp_features import box_bound,ball_bound
from state_cp_model import install_state_cp

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--qp-max-iter',type=int)
p.add_argument('--convexify',action='store_true')
p.add_argument('--outer-max-iter',type=int)
p.add_argument('--eigenvalue-relative-floor',type=float,default=1e-8)
p.add_argument('--normalize-objective',action='store_true')
p.add_argument('--qp-tolerance',type=float)
args=p.parse_args();root=args.root.resolve();source=(root/'source').resolve();out=root/'results'
suffix='' if args.qp_max_iter is None else '_iter'+str(args.qp_max_iter)
if args.convexify:suffix='_convexified'+suffix
if args.outer_max_iter is not None:suffix+='_outer'+str(args.outer_max_iter)
if args.eigenvalue_relative_floor!=1e-8:suffix+='_rf'+str(args.eigenvalue_relative_floor).replace('.','p')
if args.normalize_objective:suffix+='_normalized'
meta=json.loads((out/'doc_cp_model.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
sys.path.insert(0,str(source));Model=install_state_cp(source,out/'doc_cp_model.json')
with patch('numpy.load',side_effect=AssertionError('Old bound load forbidden')),patch('torch.load',side_effect=AssertionError('Old bound load forbidden')):
    m=Model()
assert not hasattr(m,'theta_poly') and not hasattr(m,'theta_bias')
train=np.load(root.parent/protocol['training']['source_task']/'results/calibration/diagnostic_traces.npz')
states=train['observation_states'].transpose(0,2,1).reshape(-1,m.nx)[::23]
box=box_bound(states,meta);ball=ball_bound(states,meta)
assert np.allclose(box,[float(m.doc_cp_box_fn(x)) for x in states],atol=1e-9,rtol=1e-9)
assert np.allclose(ball,[float(m.state_cp_bound_fn(x)) for x in states],atol=1e-9,rtol=1e-9)
x,v=ca.MX.sym('x',m.nx),ca.MX.sym('v',m.ny)
y=m.measurement(x,v);fn=ca.Function('doc_measurement_jacobians',[x,v],[ca.jacobian(y,v),ca.jacobian(y,x)])
for state in states[::max(1,len(states)//5)]:
    F,C=fn(state,np.zeros(m.ny))
    assert np.allclose(F,float(m.state_cp_bound_fn(state))*np.eye(m.ny)) and np.allclose(C,m.C)
corners=np.array(list(__import__('itertools').product([-1.,1.],repeat=m.ny)))
assert np.max(np.linalg.norm(corners,axis=1))<=np.sqrt(m.ny)+1e-12
checks={'old_bound_loading_blocked':True,'numeric_symbolic_box_and_ball_bounds_equal':True,
 'measurement_jacobians_checked':True,'box_corners_contained_in_ball':True,
 'ball_radius_factor':float(np.sqrt(m.ny)),'local_hessian_projection':args.convexify}
(out/('model_checks'+suffix+'.json')).write_text(json.dumps(checks,indent=2)+'\n')
from solver.SCP_OF_SLS import SCP_OF_SLS
original_check=SCP_OF_SLS.check_convergence_scp;original_solve=SCP_OF_SLS.solve
convergence=[];status={}
def check(self,*args,**kwargs):
    value=original_check(self,*args,**kwargs)
    cur=self.current_iteration_scp;X=cur['primal_x'];U=cur['primal_u']
    defects=[np.max(np.abs(np.asarray(self.m.ddyn(X[:,k],U[:,k],h=self.m.dt)).reshape(-1)-X[:,k+1])) for k in range(self.N)]
    row={'outer_index':len(convergence),'original_converged':bool(value),
      'step_norm':float(np.linalg.norm(cur.get('delta_vec',[]))),
      'max_nonlinear_nominal_dynamics_defect':float(max(defects))}
    if 'backoff' in cur:
        chi=np.vstack([X[:,:-1],U]);row['max_tightened_stage_residual']=float(np.max(np.asarray(self.m.G)@chi-self.m.g[:,None]+cur['backoff'].T))
    convergence.append(row)
    (out/('convergence_history'+suffix+'.json')).write_text(json.dumps(convergence,indent=2)+'\n')
    return value
def solve(self,*args,**kwargs):
    if globals()['args'].outer_max_iter is not None:self.MAX_ITER_SCP=globals()['args'].outer_max_iter
    result=original_solve(self,*args,**kwargs)
    current=self.current_iteration_scp
    arrays={k:np.asarray(current[k]) for k in ['primal_x','primal_u','initial_x'] if k in current}
    if arrays:np.savez_compressed(out/('planner_last_iterate'+suffix+'.npz'),**arrays)
    status.update(solver_success_flag=bool(result.get('success',False)),
      convergence_checks=len(convergence),converged=any(row['original_converged'] for row in convergence),
      maximum_outer_iterations=int(self.MAX_ITER_SCP),reported_iterations=result.get('iterations'))
    return result
SCP_OF_SLS.check_convergence_scp=check;SCP_OF_SLS.solve=solve
from qp_audit import install_qp_audit
if args.qp_max_iter is not None or args.qp_tolerance is not None:
    from solver.qp import QP
    original_initialize=QP.initialize_solver
    def initialize(self,*a,**kw):
        if args.qp_max_iter is not None:self.nominal_solver_options['osqp']['max_iter']=args.qp_max_iter
        if args.qp_tolerance is not None:self.nominal_solver_options['osqp'].update(eps_abs=args.qp_tolerance,eps_rel=args.qp_tolerance)
        return original_initialize(self,*a,**kw)
    QP.initialize_solver=initialize
install_qp_audit(out/('qp_audit'+suffix+'.json'),convexify=args.convexify,relative_floor=args.eigenvalue_relative_floor,normalize_objective=args.normalize_objective)
plan=out/('doc_cp_plan'+suffix);plan.mkdir(exist_ok=True);os.chdir(plan)
entry='expe/parking_lot/main_parking_lot.py' if meta['scene']=='parking' else 'expe/quad_perception/main_quad10d_perception.py'
filename='unicycle_of_perception_0.npz' if meta['scene']=='parking' else 'quad10d_perception_0.npz'
t=time.perf_counter()
try:
    runpy.run_path(str(source/entry),run_name='__main__')
    status['plan_exported']=True
    sys.argv=['check_plan.py','--source',str(source),'--output',str(plan),'--plan',str(plan/'data'/filename)]
    runpy.run_path(str(root/'check_plan.py'),run_name='__main__')
    d=np.load(plan/'data'/filename);nom=d['nominal_traj'].T
    status.update(nominal_box_bound_min=float(box_bound(nom,meta).min()),
       nominal_box_bound_max=float(box_bound(nom,meta).max()))
except Exception as error:
    status.update(plan_exported=False,error_type=type(error).__name__,error=str(error))
    traceback.print_exc()
finally:
    status.update(wall_seconds=time.perf_counter()-t,strict_original_solver=args.qp_max_iter is None and not args.convexify and args.outer_max_iter is None,
       qp_max_iterations_override=args.qp_max_iter,
       local_hessian_projection=args.convexify,outer_iterations_override=args.outer_max_iter,
       eigenvalue_relative_floor=args.eigenvalue_relative_floor,
       objective_normalization=args.normalize_objective,qp_tolerance=args.qp_tolerance,
       enclosure='sqrt(ny) times infinity-box bound as Euclidean-ball radius')
    (out/('plan_status'+suffix+'.json')).write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2),flush=True)
if not status.get('plan_exported'):sys.exit(2)
