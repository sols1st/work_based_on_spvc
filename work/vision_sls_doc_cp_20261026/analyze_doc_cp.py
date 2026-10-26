"""Independently audit document CP calibration, closed loop and outer-ball interface."""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from doc_cp_features import features,design,envelope,positive_scale,box_bound,ball_bound
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--variant',choices=['strict','convexified','regularized','conditioned'],default='strict')
args=p.parse_args();root=args.root.resolve();out=root/'results'
suffix={'strict':'','convexified':'_convexified_iter50000_outer80','regularized':'_convexified_iter50000_outer100_rf0p001','conditioned':'_convexified_iter100000_outer100_rf0p001_normalized'}[args.variant]
cp_phase='doc_cp'+suffix
protocol=json.loads((root/'protocol.json').read_text());scene=protocol['scene']
model=json.loads((out/'doc_cp_model.json').read_text())
filename='unicycle_of_perception_0.npz' if scene=='parking' else 'quad10d_perception_0.npz'
modelfile='dyn/unicycle_of_perception.py' if scene=='parking' else 'dyn/quad_10d_of_perception.py'
nodes=[n for n in ast.walk(ast.parse((root/'source'/modelfile).read_text())) if isinstance(n,ast.Assign) and
 any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='C' for t in n.targets)]
C=np.asarray(ast.literal_eval(nodes[0].value.args[0]))
if scene=='parking':
    xmin=np.array([-3.5,-2,-2*np.pi,-1]);xmax=np.array([.5,1.9,2*np.pi,2])
    fmin=np.array([0,.35,-2*np.pi,-1]);fmax=np.array([.45,.65,2*np.pi,1])
    umin=np.array([-np.pi,-4]);umax=np.array([np.pi,4]);target=np.array([.25,.5])
    obstacles=[([-3.55,.7],.4),([-.6,1.15],.35),([-1.2,-1.2],.4)]
    original_plan=root/'source/expe/parking_lot/data'/filename
    offset=np.array([3.,0,0,0])
else:
    xmin=np.array([-10,-10,0,-5,-5,-5,-2,-2,-2,-2]);xmax=np.array([10,10,10,5,5,5,2,2,2,2])
    fmin=xmin.copy();fmax=xmax.copy();umin=np.array([-5,-5,0]);umax=np.array([5,5,40]);target=np.array([-1.5,2.5,1.4])
    obstacles=[([1,1],.2),([-1.5,4],.2),([.5,3],.2),([-.5,1],.2),([-1,2.5],.2)]
    original_plan=root/'source/expe/quad_perception/data'/filename
    offset=np.zeros(10)
def ode(x,u,w):
    if scene=='parking':
        f=np.column_stack((x[:,3]*np.cos(x[:,2]),x[:,3]*np.sin(x[:,2]),u[:,0],u[:,1]))
    else:
        f=np.column_stack((x[:,3],x[:,4],x[:,5],9.81*np.tan(x[:,6])-3*x[:,3],
          9.81*np.tan(x[:,7])-3*x[:,4],u[:,2]-9.81-x[:,5],x[:,8]-10*x[:,6],
          x[:,9]-10*x[:,7],50*u[:,0]-10*x[:,8],50*u[:,1]-10*x[:,9]))
    return f+.01*w

def metrics(phase):
    d=np.load(out/phase/'diagnostic_traces.npz')
    summary=json.loads((out/phase/'summary.json').read_text())
    X,U,Y=d['states'],d['inputs'],d['observations'];R,nx,nsteps=X.shape;T=nsteps-1;nu=U.shape[1];ny=Y.shape[1]
    assert R==50 and summary['seed']==protocol['evaluation']['seed']
    for key in d.files:assert np.isfinite(d[key]).all(),(phase,key)
    indices=np.r_[0,np.arange(T-1)]
    aligned=X[:,:,indices]
    assert np.array_equal(aligned,d['observation_states'])
    residual=Y-np.einsum('ab,nbt->nat',C,aligned+offset[None,:,None])
    residual_error=float(np.max(np.abs(residual-d['residuals'])))
    assert residual_error<1e-10
    plan=np.load(original_plan if phase=='baseline' else out/('doc_cp_plan'+suffix)/'data'/filename)
    assert np.array_equal(plan['nominal_traj'],d['nominal_states'])
    K=plan['Phi_uy']-plan['Phi_ux']@np.linalg.solve(plan['Phi_xx'],plan['Phi_xy'])
    innovation=Y-(C@(plan['nominal_traj'][:,indices]+offset[:,None]))[None]
    reconstructed=(innovation.transpose(0,2,1).reshape(R,T*ny)@K[:T*nu,:T*ny].T).reshape(R,T,nu).transpose(0,2,1)+plan['nominal_input'][None]
    control_error=float(np.max(np.abs(U-reconstructed)));assert control_error<1e-8
    xx=X[:,:,:-1].transpose(0,2,1).reshape(-1,nx)
    uu=U.transpose(0,2,1).reshape(-1,nu)
    ww=d['disturbances'].transpose(0,2,1).reshape(-1,nx)
    k1=ode(xx,uu,ww);k2=ode(xx+.075*k1,uu,ww);k3=ode(xx+.075*k2,uu,ww);k4=ode(xx+.15*k3,uu,ww)
    rk=xx+.15/6*(k1+2*k2+2*k3+k4)
    dynamics_error=float(np.max(np.abs(rk-X[:,:,1:].transpose(0,2,1).reshape(-1,nx))))
    assert dynamics_error<1e-8
    path=((X[:,:,:-1]>=xmin[None,:,None]-1e-3)&(X[:,:,:-1]<=xmax[None,:,None]+1e-3)).all(axis=(1,2))
    terminal=((X[:,:,-1]>=fmin[None]-1e-3)&(X[:,:,-1]<=fmax[None]+1e-3)).all(axis=1)
    inputs=((U>=umin[None,:,None]-1e-3)&(U<=umax[None,:,None]+1e-3)).all(axis=(1,2))
    xy=X[:,:2,:].transpose(0,2,1)
    clear=[];hit=np.zeros(R,dtype=bool)
    for center,radius in obstacles:
        d2=np.sum((xy-center)**2,axis=2);clear.append(np.sqrt(d2)-radius)
        hit|=(d2<=radius**2+1e-4).any(axis=1)
    arrive=((X[:,:len(target),-1]>=fmin[None,:len(target)]-1e-3)&(X[:,:len(target),-1]<=fmax[None,:len(target)]+1e-3)).all(axis=1)
    dist=np.linalg.norm(X[:,:len(target),-1]-target,axis=1)
    state_half=np.vstack((plan['backoff'][:,:nx],plan['backoff_f'][None,:nx]))
    input_half=plan['backoff'][:,nx:nx+nu]
    sx=np.abs(X-plan['nominal_traj'][None])-state_half.T[None]
    ux=np.abs(U-plan['nominal_input'][None])-input_half.T[None]
    norms=np.linalg.norm(residual,axis=1)
    states=aligned.transpose(0,2,1).reshape(-1,nx)
    newbox=box_bound(states,model).reshape(R,T)
    newb=ball_bound(states,model).reshape(R,T)
    if phase.startswith('doc_cp'):
        captured=d['error_bounds'] if scene=='parking' else d['error_bounds'][:,0,:]
        assert np.allclose(newb,captured,atol=1e-10)
    own=(norms<=d['error_bounds']+1e-8) if scene=='parking' else (np.linalg.norm(residual/d['error_bounds'],axis=1)<=1+1e-8)
    covered=np.max(np.abs(residual),axis=1)<=newbox+1e-8;ball_covered=norms<=newb+1e-8;split=protocol['evaluation']['axis_stress_rollouts']
    assert np.all(~covered|ball_covered)
    f=features(states,scene)
    outside=((f<np.array(model['feature_train_min'])-1e-8)|(f>np.array(model['feature_train_max'])+1e-8)).any(axis=1)
    result={'author_successes':int((arrive&~hit).sum()),'collision_rollouts':int(hit.sum()),
     'full_state_input_collision_constraint_violations':int((~(path&terminal&inputs&~hit)).sum()),
     'stage_state_constraint_violating_rollouts':int((~path).sum()),
     'terminal_state_constraint_violating_rollouts':int((~terminal).sum()),
     'input_constraint_violating_rollouts':int((~inputs).sum()),
     'max_stage_state_bound_excess_by_component':np.maximum(X[:,:,:-1]-xmax[None,:,None],xmin[None,:,None]-X[:,:,:-1]).max(axis=(0,2)).tolist(),
     'max_input_bound_excess_by_component':np.maximum(U-umax[None,:,None],umin[None,:,None]-U).max(axis=(0,2)).tolist(),
     'minimum_sampled_obstacle_clearance_m':float(np.min(clear)),
     'terminal_position_distance_mean_m':float(dist.mean()),'terminal_position_distance_max_m':float(dist.max()),
     'state_tube_violating_rollouts_tolerance_1e-6':int((sx>1e-6).any(axis=(1,2)).sum()),
     'input_tube_violating_rollouts_tolerance_1e-6':int((ux>1e-6).any(axis=(1,2)).sum()),
     'max_state_tube_excess':float(sx.max()),'max_input_tube_excess':float(ux.max()),
     'own_bound_l2_point_coverage':float(own.mean()),'own_bound_l2_trajectories_covered':int(own.all(axis=1).sum()),
     'doc_box_point_coverage':float(covered.mean()),
     'doc_box_trajectories_covered':int(covered.all(axis=1).sum()),
     'doc_box_random_trajectories_covered':int(covered[split:].all(axis=1).sum()),
     'doc_box_stress_trajectories_covered':int(covered[:split].all(axis=1).sum()),
     'doc_box_min':float(newbox.min()),'doc_box_max':float(newbox.max()),
     'feedback_operator_frobenius_norm':float(np.linalg.norm(K)),
     'raw_trajectory_residual_l2_max':float(norms.max()),
     'raw_trajectory_residual_infinity_max':float(np.max(np.abs(residual))),
     'unclipped_envelope_min':float(envelope(states,model).min()),
     'unclipped_envelope_max':float(envelope(states,model).max()),
     'uncovered_box_observations':int((~covered).sum()),
     'uncovered_box_observations_at_scale_floor':int(((~covered)&(positive_scale(states,model).reshape(R,T)<=model['epsilon']+1e-12)).sum()),
     'feature_outside_training_coordinate_ranges_fraction':float(outside.mean()),
     'outer_ball_point_coverage':float(ball_covered.mean()),'outer_ball_trajectories_covered':int(ball_covered.all(axis=1).sum()),
     'outer_ball_min':float(newb.min()),'outer_ball_max':float(newb.max()),
     'positive_scale_floor_fraction':float(np.mean(positive_scale(states,model)<=model['epsilon']+1e-12)),
     'independent_residual_reconstruction_error':residual_error,
     'independent_feedback_reconstruction_error':control_error,'independent_rk4_transition_error':dynamics_error}
    for key in ['author_successes','collision_rollouts','full_state_input_collision_constraint_violations',
                'minimum_sampled_obstacle_clearance_m','state_tube_violating_rollouts_tolerance_1e-6',
                'input_tube_violating_rollouts_tolerance_1e-6']:
        assert np.isclose(result[key],summary[key],atol=1e-9),(phase,key,result[key],summary[key])
    return result,d,newbox,np.max(np.abs(residual),axis=1)


cal=np.load(out/'calibration/diagnostic_traces.npz');n,_,T=cal['residuals'].shape
cs=cal['observation_states'].transpose(0,2,1).reshape(-1,cal['states'].shape[1])
cal_norm=np.max(np.abs(cal['residuals']),axis=1)
scores=(cal_norm/positive_scale(cs,model).reshape(n,T)).max(axis=1)
k=math.ceil((n+1)*(1-protocol['alpha']));q=float(np.sort(scores)[k-1])
assert abs(q-model['cp_multiplier'])<1e-10
assert np.allclose(scores,np.load(out/'doc_cp_scores.npz')['trajectory_scores'])
train=np.load(root.parent/protocol['training']['source_task']/'results/calibration/diagnostic_traces.npz')
tr=np.max(np.abs(train['residuals']),axis=1).reshape(-1)
ts=train['observation_states'].transpose(0,2,1).reshape(-1,train['states'].shape[1])
fit=np.load(out/'training_fit.npz');pred=envelope(ts,model);xi=fit['slack']
assert np.allclose(pred,fit['envelope']) and np.allclose(tr,fit['residual_infinity'])
assert max(float(np.max(tr-pred-xi)),float(-pred.min()),float(-xi.min()))<1e-5
objective=pred.mean()+model['gamma']*np.sum(np.asarray(model['coefficients'])**2)+model['mu']*xi.mean()
assert abs(objective-model['objective_value'])<1e-6
b,bd,bb,bnorm=metrics('baseline');datasets=[('baseline',bd,bb,bnorm,'#3478b5')]
methods={'baseline':b};paired=None
if (out/cp_phase/'summary.json').exists():
    c,cd,cb,cnorm=metrics(cp_phase);assert np.array_equal(bd['disturbances'],cd['disturbances'])
    methods['doc_cp']=c;paired=True;datasets.append(('doc_cp',cd,cb,cnorm,'#df8130'))
plan_checks=out/('doc_cp_plan'+suffix)/'plan_checks.json'
result={'protocol':protocol,'variant':args.variant,'model':model,'methods':methods,'paired_disturbances_identical':paired,
 'calibration_scores_and_multiplier_independently_equal':True,'training_objective_and_constraints_independently_equal':True,
 'plan_status':json.loads((out/('plan_status'+suffix+'.json')).read_text()),
 'model_checks':json.loads((out/('model_checks'+suffix+'.json')).read_text()),
 'plan_checks':json.loads(plan_checks.read_text()) if plan_checks.exists() else None,
 'traces_sha256':{s:hashlib.sha256((out/(cp_phase if s=='doc_cp' else s)/'diagnostic_traces.npz').read_bytes()).hexdigest() for s in ['calibration',*methods]}}
if args.variant!='strict':result['numerical_adaptation_protocol']=json.loads((root/({'regularized':'numerical_adaptation_regularized_protocol.json','conditioned':'numerical_adaptation_conditioned_protocol.json'}.get(args.variant,'numerical_adaptation_protocol.json'))).read_text())
fig,axes=plt.subplots(1,3,figsize=(16,5))
for name,d,bounds,norms,color in datasets:
    for x in d['states']:axes[0].plot(x[0],x[1],c=color,alpha=.2,lw=.7)
    axes[0].plot(d['nominal_states'][0],d['nominal_states'][1],c=color,label=name,lw=2)
    ratio=(norms/bounds).max(axis=1)
    axes[1].plot(np.sort(ratio),np.arange(1,51)/50,c=color,label=name)
    axes[2].plot(bounds.mean(axis=0),c=color,label=name)
for center,radius in obstacles:axes[0].add_patch(plt.Circle(center,radius,color='#d96363',alpha=.5))
axes[0].scatter(*target[:2],marker='*',color='black',s=100)
axes[0].set_aspect('equal',adjustable='datalim');axes[0].set(xlabel='x (m)',ylabel='y (m)',title='Paired visual trajectories' if len(datasets)>1 else 'Reference-policy visual trajectories')
axes[1].axvline(1,c='black',ls='--');axes[1].set_xscale('log')
axes[1].set(xlabel='Max infinity residual / CP box bound',ylabel='Cumulative fraction',title='Document box trajectory coverage')
axes[2].set(xlabel='Observation step',ylabel='Mean CP box half-width',title='Multiplicative box along each policy')
for ax in axes:ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(out/('doc_cp_comparison'+suffix+'.png'),dpi=180);plt.close(fig)
(out/('doc_cp_comparison'+suffix+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
