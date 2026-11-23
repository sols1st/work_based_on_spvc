"""Fresh paired camera confirmation of preselected first-seed SPVT training.

Original nominal road renderer and frozen DINO patch readout; no frozen image
replay. Each arm obtains its own new 199-trajectory threshold before its 100
holdout episodes. This extends, rather than changes, the previous synthetic run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from .tasks import RESULTS,load_task

CAL_SEED=202611188501
TEST_SEED=202611188502
SELECTED_TRAINING_SEED=202611188101


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def metrics(data):
    state=data['states'];u=.45*data['actions'];error=data['errors']
    return dict(mean_abs_e_m=np.abs(state[:,1:,0]).mean(1),peak_abs_e_m=np.abs(state[:,:,0]).max(1),
        mean_abs_heading_rad=np.abs(state[:,1:,1]).mean(1),mean_steering_change_rad=np.abs(np.diff(u,axis=1)).mean(1),
        mean_abs_steering_rad=np.abs(u).mean(1),trajectory_error_score=np.abs(error).max(axis=(1,2)),unsafe=data['unsafe'])


def run(root,device,batch,budget):
    import torch
    from Aebs.lane_keeping.robust_trajectory_cp import main as collect_main
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    original=RESULTS/'lane_robust_synthesis_20261116_v2/package'
    trained=RESULTS/'paper_controller_comparison_20261118_v1'/str(SELECTED_TRAINING_SEED)/'spvt/actor.pt'
    # RESULTS is artical-F122/results; project external is two parents above it.
    repo=RESULTS.parent.parent/'external/dinov2'
    weights=RESULTS.parent.parent/'external/dinov2_weights/dinov2_vits14_pretrain.pth'
    protocol=dict(date='2026-11-18',arms=['anchor','spvt_first_seed'],selected_training_seed=SELECTED_TRAINING_SEED,
        selection='first prespecified training seed; no selection using this new calibration or test',
        camera='same original live road renderer, new image rendered after every state update; not frozen replay',
        new_calibration_per_arm=199,new_holdout_per_arm=100,horizon=80,alpha=.05,
        calibration_seed=CAL_SEED,holdout_seed=TEST_SEED,paired_external_scenarios=True,
        physical_units='e metres, heading radians; logged policy action multiplied by .45 for steering radians',
        device=device,batch_size=batch,dtype='float32 DINO/readout/actor; float64 plant',
        gpu_allocator_fraction_limit=.035,calibration_rank=190,threads=4,
        effective_new_image_observations=2*(199+100)*80,
        frozen_before_collection=dict(anchor_actor=digest(original/'actor.pt'),spvt_actor=digest(trained),
             observer=digest(original/'patch_projection.pt'),barrier=digest(original/'sbc.pt'),dino=digest(weights)),
        wrapper_source_sha256=digest(__file__),old_entrypoint_sha256=digest(Path(__file__).parents[1]/'lane_keeping/robust_trajectory_cp.py'),
        note='Legacy per-arm protocol hard-codes an older date; this wrapper records actual experiment date and registration.',
        guarantee='new per-policy trajectory CP then fresh robust B audit; confidence and certificate-selection qualifications remain',
        inference_scope='one selected controller pair; no claim over unseen training seeds or real-road cameras')
    save(root/'protocol.json',protocol);shutil.copyfile(__file__,root/'source.py')
    torch.cuda.set_device(torch.device(device));torch.cuda.set_per_process_memory_fraction(.035,torch.device(device))
    started=time.monotonic()
    for arm,actor in [('anchor',original/'actor.pt'),('spvt_first_seed',trained)]:
        package=root/arm/'package';package.mkdir(parents=True)
        for name,source in [('actor.pt',actor),('sbc.pt',original/'sbc.pt'),('patch_projection.pt',original/'patch_projection.pt')]:shutil.copyfile(source,package/name)
        at=time.monotonic()
        collect_main(['--package',str(package),'--output-dir',str(root/arm/'cp'),'--dino-repo',str(repo),'--dino-weights',str(weights),
            '--calibration-seed',str(CAL_SEED),'--holdout-seed',str(TEST_SEED),'--batch-size',str(batch),'--device',device])
        torch.cuda.empty_cache()
        save(root/arm/'collection_runtime.json',dict(seconds=time.monotonic()-at,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(torch.device(device)),
             peak_reserved_bytes=torch.cuda.max_memory_reserved(torch.device(device))))
        # This shadow has the NEW threshold from this arm, never the old q.
        shadow=root/arm/'shadow/results/lane_robust_synthesis_20261116_v2'
        (shadow/'package').mkdir(parents=True);(shadow/'final_cp').mkdir()
        for name in ['actor.pt','sbc.pt']:shutil.copyfile(package/name,shadow/'package'/name)
        shutil.copyfile(root/arm/'cp/calibration.json',shadow/'final_cp/calibration.json')
        task=load_task('lane',root=root/arm/'shadow');torch.set_num_threads(1)
        audit=task.current_verify(budget);audit['metadata']=task.metadata
        audit['new_trajectory_calibration_sha256']=digest(root/arm/'cp/calibration.json')
        save(root/arm/'current_new_cp.json',audit)
        print(json.dumps(dict(arm=arm,new_q=float(task.error_hi[0]),fresh_B_status=audit.get('status'))),flush=True)
    records={arm:dict(np.load(root/arm/'cp/holdout.npz',allow_pickle=False)) for arm in protocol['arms']}
    a,b=(records[arm] for arm in protocol['arms'])
    if not np.array_equal(a['episode_seeds'],b['episode_seeds']) or not np.array_equal(a['kappas'],b['kappas']) or not np.array_equal(a['states'][:,0],b['states'][:,0]):raise AssertionError('paired scenario mismatch')
    metric={arm:metrics(records[arm]) for arm in protocol['arms']}
    rng=np.random.default_rng(202611188503);indices=rng.integers(100,size=(5000,100));comparison={};rows=[]
    for key in metric['anchor']:
        aa=np.asarray(metric['anchor'][key],float);bb=np.asarray(metric['spvt_first_seed'][key],float)
        delta=bb-aa
        comparison[key]=dict(anchor_mean=float(aa.mean()),spvt_mean=float(bb.mean()),spvt_minus_anchor=float(delta.mean()),
            paired_trajectory_ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
            spvt_lower_trajectories=int((delta<0).sum()),spvt_higher_trajectories=int((delta>0).sum()))
    for i in range(100):
        row=dict(trajectory=i,episode_seed=int(a['episode_seeds'][i]),curvature=float(a['kappas'][i]))
        for arm in protocol['arms']:
            for key,value in metric[arm].items():row[arm+'__'+key]=float(value[i])
        rows.append(row)
    with (root/'paired_holdout_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    reports={arm:json.loads((root/arm/'cp/report.json').read_text()) for arm in protocol['arms']}
    audits={arm:json.loads((root/arm/'current_new_cp.json').read_text()) for arm in protocol['arms']}
    save(root/'summary.json',dict(protocol_sha256=digest(root/'protocol.json'),paired_metrics=comparison,
        per_arm={arm:dict(q=reports[arm]['calibration']['radius'],covered=reports[arm]['holdout']['coverage']['covered'],
          unsafe=reports[arm]['holdout']['unsafe_episodes'],fresh_barrier_passed=audits[arm].get('passed',False),
          fresh_barrier_status=audits[arm].get('status'),verification_seconds=audits[arm]['elapsed_seconds']) for arm in protocol['arms']},
        elapsed_seconds=time.monotonic()-started,
        statistical_unit='paired complete new image trajectories, 100 holdout; fixed selected controller pair',
        caution='CP is marginal over calibration and a new trajectory; selecting only verified thresholds does not automatically give conditional 95% safety. Verification is floating-point without directed rounding.'))
    print(json.dumps(dict(stage='complete',output=str(root/'summary.json'))),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESULTS/'paper_controller_image_confirmation_20261118_v1')
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch',type=int,default=8);p.add_argument('--budget',type=float,default=300.)
    a=p.parse_args();run(a.root,a.device,a.batch,a.budget)


if __name__=='__main__':main()
