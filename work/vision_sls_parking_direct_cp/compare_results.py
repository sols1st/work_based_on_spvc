"""Compare paired baseline/CP runs at the original code's evaluation level."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
from matplotlib import pyplot as plt

root=Path(__file__).resolve().parent
out=root/'results'
protocol=json.loads((root/'protocol.json').read_text())
cp=json.loads((out/'direct_cp_calibration.json').read_text())
baseline=np.load(out/'baseline/diagnostic_traces.npz')
direct=np.load(out/'direct_cp/diagnostic_traces.npz')
assert baseline['states'].shape==direct['states'].shape==(50,4,31)
assert np.array_equal(baseline['disturbances'],direct['disturbances']), 'Paired disturbance mismatch'
assert np.all(direct['error_bounds']==cp['radius']), 'CP bound must be constant'
original_plan=np.load(root/'source/expe/parking_lot/data/unicycle_of_perception_0.npz')
assert np.array_equal(original_plan['nominal_traj'],baseline['nominal_states'])
assert np.array_equal(original_plan['nominal_input'],baseline['nominal_inputs'])
result={'host':platform.node(),'protocol':protocol,'calibration':cp,
        'paired_disturbances_identical':True,'methods':{},
        'nominal_plan_changed':not np.array_equal(baseline['nominal_states'],direct['nominal_states']),
        'source_checks':{},'scope':'Same source implementation and empirical evaluation protocol. '
        'The direct bound is calibrated on raw residuals, not a correction to the old polynomial. '
        'CP reference-policy coverage and replanned-policy empirical coverage are reported separately.'}
fig,axes=plt.subplots(1,3,figsize=(16,5))
for name,data,color in [('baseline',baseline,'#3478b5'),('direct_cp',direct,'#df8130')]:
    summary=json.loads((out/name/'summary.json').read_text())
    for key in ['states','inputs','observations','residuals','error_bounds']:
        assert np.isfinite(data[key]).all(),(name,key)
    norms=np.linalg.norm(data['residuals'],axis=1)
    scores=norms.max(axis=1)
    bounds=data['error_bounds']
    own=norms<=bounds+1e-8
    metric={k:summary[k] for k in ['author_successes','author_constraint_violations',
             'full_state_input_collision_constraint_violations','collision_rollouts',
             'minimum_sampled_obstacle_clearance_m',
             'state_tube_violating_rollouts_tolerance_1e-6',
             'input_tube_violating_rollouts_tolerance_1e-6',
             'max_state_tube_excess','max_input_tube_excess','wall_seconds']}
    distances=np.linalg.norm(data['states'][:,:2,-1]-np.array([.25,.5]),axis=1)
    metric.update(terminal_position_distance_mean_m=float(distances.mean()),
        terminal_position_distance_max_m=float(distances.max()),raw_trajectory_l2_score_max=float(scores.max()),
        own_bound_l2_point_coverage=float(own.mean()),
        own_bound_l2_trajectories_covered=int(own.all(axis=1).sum()),
        own_bound_l2_random_trajectories_covered=int(own[8:].all(axis=1).sum()),
        own_bound_l2_stress_trajectories_covered=int(own[:8].all(axis=1).sum()),
        direct_cp_radius_point_coverage=float(np.mean(norms<=cp['radius']+1e-12)),
        direct_cp_radius_trajectories_covered=int(np.sum(scores<=cp['radius']+1e-12)),
        direct_cp_radius_random_trajectories_covered=int(np.sum(scores[8:]<=cp['radius']+1e-12)))
    metric.update(error_bound_min=float(bounds.min()),error_bound_max=float(bounds.max()),
        maximum_state_tube_halfwidth=float(data['state_halfwidth'].max()),
        maximum_input_tube_halfwidth=float(data['input_halfwidth'].max()),
        terminal_speed_abs_mean=float(np.abs(data['states'][:,3,-1]).mean()))
    plan_path=(root/'source/expe/parking_lot/data/unicycle_of_perception_0.npz'
               if name=='baseline' else out/'direct_cp_plan/data/unicycle_of_perception_0.npz')
    plan=np.load(plan_path)
    K=plan['Phi_uy']-plan['Phi_ux']@np.linalg.solve(plan['Phi_xx'],plan['Phi_xy'])
    metric['feedback_operator_frobenius_norm']=float(np.linalg.norm(K))
    metric['feedback_operator_spectral_norm']=float(np.linalg.norm(K,2))
    result['methods'][name]=metric
    for path in data['states']:
        axes[0].plot(path[0],path[1],color=color,alpha=.16,lw=.7)
    axes[0].plot(data['nominal_states'][0],data['nominal_states'][1],color=color,lw=2,label=name)
    axes[1].plot(np.sort(scores),np.arange(1,51)/50,color=color,lw=2,label=name)
    axes[2].plot(np.arange(31)*.15,data['states'][:,3,:].mean(axis=0),color=color,lw=2,label=name)
obstacles=[([-3.55,.7],.4),([-.6,1.15],.35),([-1.2,-1.2],.4)]
for center,radius in obstacles:
    axes[0].add_patch(plt.Circle(center,radius,color='#d96363',alpha=.5))
axes[0].scatter(.25,.5,marker='*',color='black',s=100,label='Target')
axes[0].set(xlabel='x (m)',ylabel='y (m)',title='Paired visual closed-loop runs')
axes[0].set_aspect('equal',adjustable='datalim')
axes[1].axvline(cp['radius'],color='black',ls='--',label='Direct CP radius')
axes[1].set(xlabel='Max raw L2 residual per trajectory',ylabel='Empirical cumulative fraction',title='Whole-trajectory errors')
axes[2].axhline(0,color='black',ls='--',label='Target speed')
axes[2].set(xlabel='Time (s)',ylabel='Mean speed (m/s)',title='Speed')
for ax in axes:
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out/'direct_cp_comparison.png',dpi=180)
plt.close(fig)
checks=json.loads((out/'direct_cp_plan/plan_checks.json').read_text())
result['direct_cp_plan_checks']=checks
result['source_checks']['adapter_sha256']=hashlib.sha256((root/'direct_cp_model.py').read_bytes()).hexdigest()
result['source_checks']['protocol_sha256']=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
result['source_checks']['original_model_no_polynomial_read']=json.loads((out/'model_checks.json').read_text())
(out/'direct_cp_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
