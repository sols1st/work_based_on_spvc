"""Original task evaluation of the belief-preserving backup governor."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import ACTION_HIGH,INITIAL_STATES,GOAL_LOW,GOAL_HIGH
from .ppo_env import ParkingPPOEnv,FrozenImageSensor
from .planned_env import perturbed_initial_state
from .planning_feedback import ReferenceBank
from .backup_governor import BackupGovernor
from .train_migration import load_ppo
from .train_ppo import sha


def evaluate(bank,policy,error_model,sensor=None,bound=.04,jitter=0.,seed=701,library=None,startup_model=None):
    env=ParkingPPOEnv(image_sensor=sensor);rows=[];paths=[];commands=[];observations=[];beliefs=[];masks=[];estimates=[];references=[]
    for j in range(15):
        initial=perturbed_initial_state(j,np.random.default_rng(seed+j),jitter)
        env.reset(options=dict(scenario=j,state=initial))
        governor=BackupGovernor(bank,policy,error_model,initial_jitter=bound,library=library,startup_model=startup_model)
        goal=(GOAL_LOW[j,:2]+GOAL_HIGH[j,:2])/2
        governor.reset(env.measurements[-1],goal)
        def full_box():
            full=np.full((2,15,4),np.nan);full[:,governor.hypotheses,:]=np.stack(governor.box)
            return full
        xs=[initial.copy()];us=[];ys=[env.measurements[-1].copy()];boxes=[full_box()]
        initial_hypotheses=governor.hypotheses.copy()
        true_hypothesis=np.flatnonzero(governor.hypotheses==j)
        contained=bool(len(true_hypothesis))
        es=[governor.controller.estimate.copy()]
        for t in range(30):
            u=governor.action();_,_,_,_,info=env.step(u/ACTION_HIGH)
            governor.receive(env.measurements[-1]);x=env.task.state
            matching=np.flatnonzero(governor.hypotheses==j)
            contained=contained and bool(len(matching))
            if len(matching):
                idx=int(matching[0]);contained=contained and bool(((x>=governor.box[0][idx])&(x<=governor.box[1][idx])).all())
            es.append(governor.controller.estimate.copy())
            xs.append(x.copy());us.append(info['action_applied']);ys.append(env.measurements[-1].copy());boxes.append(full_box())
        row=dict(scenario=j,success=info['episode_success'],collision=info['collision_seen'],
                 out_of_bounds=info['out_of_bounds_seen'],final_goal_distance=info['final_goal_distance'],
                 hypotheses=governor.hypotheses.tolist(),interventions=governor.interventions,
                 initial_hypotheses=initial_hypotheses.tolist(),
                 startup_true_hypothesis_present=bool(len(true_hypothesis)),
                 selected_reference=int(governor.controller.reference),
                 invalid_measurements=governor.invalid_measurements,invalid_proposals=governor.invalid_proposals,
                 backup_reference=governor.backup_reference,
                 goal_backup_certified=governor.goal_backup_certified,
                 minimum_blend=governor.minimum_blend,physical_trajectory_contained=contained)
        print(json.dumps(row),flush=True);rows.append(row);paths.append(xs);commands.append(us);observations.append(ys)
        estimates.append(es);references.append(governor.controller.reference)
        beliefs.append(np.asarray(boxes));masks.append(np.isfinite(np.asarray(boxes)[:,0,:,0]))
    env.close()
    return (dict(episodes=15,success_count=sum(r['success'] for r in rows),
                collision_count=sum(r['collision'] for r in rows),out_of_bounds_count=sum(r['out_of_bounds'] for r in rows),
                mean_final_goal_distance=float(np.mean([r['final_goal_distance'] for r in rows])),
                all_physical_trajectories_contained=all(r['physical_trajectory_contained'] for r in rows),per_scenario=rows),
                dict(states=np.asarray(paths),actions=np.asarray(commands),observations=np.asarray(observations),
                     estimates=np.asarray(estimates),references=np.asarray(references),
                     belief_boxes=np.asarray(beliefs),hypothesis_masks=np.asarray(masks)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--error-model')
    p.add_argument('--output-dir',required=True);p.add_argument('--sensor',choices=['nominal','surrogate','images'],default='nominal')
    p.add_argument('--bound',type=float,default=.04);p.add_argument('--jitter',type=float,default=0.)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--backup-library')
    p.add_argument('--startup-model');p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();out=Path(a.output_dir);root=Path(a.candidate)
    if out.exists() or not 0<=a.jitter<=a.bound:p.error('new output and 0 <= jitter <= initial bound required')
    if not a.error_model and not a.startup_model:p.error('startup calibration or legacy error model required')
    torch.set_num_threads(a.threads);out.mkdir(parents=True)
    error={} if not a.error_model else json.loads(Path(a.error_model).read_text())
    if error and error['provenance']['projection_sha256']!=sha(root/'projection.pt'):raise ValueError('projection mismatch')
    bank=ReferenceBank(root/'references.npz');policy=load_ppo(root/'ppo.zip',device='cpu');sensor=None
    cfg=dict(vars(a),ppo_sha256=sha(root/'ppo.zip'),reference_sha256=sha(root/'references.npz'),
             projection_sha256=sha(root/'projection.pt'),error_model_sha256=None if not a.error_model else sha(a.error_model),
             process_noise='zero',numerical_allowance_per_step=1e-12,
             startup_model_sha256=None if not a.startup_model else sha(a.startup_model),
             backup_library_sha256=None if not a.backup_library else sha(a.backup_library),
             posterior_pruning='none after startup' if a.startup_model else 'delayed CP compatibility; domain-crossing boxes retained',
             safety_certificate_status='structural proof and implementation audit pending')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');started=time.monotonic()
    if a.sensor=='images':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sensor=FrozenImageSensor('../VISION-SLS','../external/dinov2','../external/dinov2_weights/dinov2_vitg14_pretrain.pth',root/'projection.pt',device=a.device)
    elif a.sensor=='surrogate':
        from .residual_replay import ResidualReplaySensor
        sensor=ResidualReplaySensor('results/vision_sls_parking_20261103_residual_data_v1/samples.npz',nominal_fraction=0)
    try:
        library=None if not a.backup_library else dict(np.load(a.backup_library))
        startup=None if not a.startup_model else json.loads(Path(a.startup_model).read_text())
        if startup is not None and startup['projection_sha256']!=cfg['projection_sha256']:raise ValueError('startup projection mismatch')
        metrics,data=evaluate(bank,policy,error,sensor,a.bound,a.jitter,a.seed,library,startup)
    finally:
        if sensor is not None and hasattr(sensor,'close'):sensor.close()
    np.savez_compressed(out/'trajectories.npz',**data)
    metrics.update(config=cfg,runtime_seconds=time.monotonic()-started)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print({k:v for k,v in metrics.items() if k not in ('config','per_scenario')},flush=True)


if __name__=='__main__':main()
