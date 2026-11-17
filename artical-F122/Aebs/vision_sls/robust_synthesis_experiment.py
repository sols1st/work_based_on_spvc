"""Isolated actor/barrier robust synthesis and fresh final CP, two legacy tasks.

Existing PPO actors initialize interval-adversarial optimization. Barrier
training starts from random weights; old barriers guide the first actor stage
only. Development tubes are not statistical contracts for the changed actor.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from .run import sha256


def write(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['lane','aebs'],required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true',help='Resume completed stages of a development run; incomplete output directories must first be archived')
    p.add_argument('--aebs-rounds',type=int,default=6)
    a=p.parse_args()
    if a.aebs_rounds<1: p.error("positive synthesis budget required")
    a.output.mkdir(parents=True,exist_ok=a.resume)
    out=a.output;source=a.source
    events=json.loads((out/'status.json').read_text()) if a.resume and (out/'status.json').exists() else []
    protocol=dict(task=a.task,source=str(source),date='2026-11-16',
        development_radius=.10 if a.task=='lane' else .45,
        actor_initialization='existing PPO checkpoint; interval robust fine-tuning, not new PPO interaction training',
        barrier_initialization='random for first barrier stage; subsequent rounds may resume',
        frontend='frozen original linear version; no nonlinear proxy',
        final_cp='fresh independent streams after freezing final actor, never inherit old actor CP',
        online_filter=False,device=a.device,
        bindings={n:sha256(source/n) for n in (['actor.pt','sbc.pt','patch_projection.pt'] if a.task=='lane'
                    else ['policy.pt','sbc.pt','observation.pt','observations.npz'])})
    if a.resume and (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text())!=protocol: raise ValueError('resume protocol changed')
    else: write(out/'protocol.json',protocol)
    def run(name,module,args):
        if any(e['stage']==name and e['state']=='complete' for e in events): return
        command=[sys.executable,'-m',module,*map(str,args)]
        for event in events:
            if event['stage']==name and event['state']=='running': event['state']='interrupted'
        events.append(dict(stage=name,state='running',command=command));write(out/'status.json',events)
        started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
        events[-1].update(state='complete' if result.returncode==0 else 'error',
                          returncode=result.returncode,seconds=time.monotonic()-started)
        write(out/'status.json',events)
        if result.returncode: raise RuntimeError('stage failed: '+name)

    if a.task=='lane':
        run('actor_robust','Aebs.lane_keeping.refine_actor',[
            '--actor',source/'actor.pt','--sbc',source/'sbc.pt','--output-dir',out/'actor_robust',
            '--radius','.10','--rounds','6','--steps-per-round','100','--seed','161','--task-weight','.01'])
        run('synthesis','Aebs.lane_keeping.synthesize',[
            '--actor',out/'actor_robust/actor.pt','--output-dir',out/'synthesis',
            '--radius','.10','--epochs','4000','--rounds','4','--seed','41'])
        dev=json.loads((out/'synthesis/report.json').read_text())
        if not dev['passed']:
            write(out/'report.json',dict(status='development_not_certified',synthesis=dev));return
        package=out/'package';package.mkdir()
        for n in ['actor.pt','sbc.pt']: shutil.copy2(out/'synthesis/verified'/n,package/n)
        shutil.copy2(source/'patch_projection.pt',package/'patch_projection.pt')
        run('final_cp','Aebs.lane_keeping.robust_trajectory_cp',[
            '--package',package,'--output-dir',out/'final_cp','--device',a.device,'--batch-size','8',
            '--dino-repo','../external/dinov2','--dino-weights','../external/dinov2_weights/dinov2_vits14_pretrain.pth',
            '--calibration-seed','202611160101','--holdout-seed','202611160102'])
        cp=json.loads((out/'final_cp/calibration.json').read_text());q=cp['radius']
        if q is None:
            write(out/'report.json',dict(status='infinite_cp_radius',cp=cp));return
        run('final_audit','Aebs.lane_keeping.verify_learned',[
            '--actor',package/'actor.pt','--sbc',package/'sbc.pt','--output-dir',out/'final_audit',
            '--radius',q,'--grid','40','--depth','18','--error-splits','8','--method','affine','--stop-on-witness'])
        audit=json.loads((out/'final_audit/certificate.json').read_text());passed=audit['passed']
        changed=sha256(package/'actor.pt')!=sha256(source/'actor.pt')
    else:
        base=out/'development_source';base.mkdir(exist_ok=a.resume)
        for n in ['policy.pt','observation.pt','observations.npz','sbc.pt','config.json']:
            shutil.copy2(source/n,base/n)
        data=np.load(base/'observations.npz',allow_pickle=False)
        np.savez(base/'splits.npz',train=np.flatnonzero(data['split']=='train'))
        contract=json.loads((source/'perception_contract.json').read_text())
        contract.update(coefficients=[1.]*4,envelope_units='m',score_units='m',
            calibration=dict(q_hat=.45),coverage={},independent_trajectory_calibration=False,
            statistical_status='development-only uncertainty design; no CP guarantee for updated actor',
            formal_image_certificate=False,source_policy_sha256=sha256(base/'policy.pt'),
            source_observation_sha256=sha256(base/'observation.pt'))
        write(base/'perception_contract.json',contract)
        actor_base=base;barrier_base=base;resume=None;dev_pass=False
        for k in range(a.aebs_rounds):
            actor_dir=out/f'actor_{k:02d}'
            run(f'actor_{k:02d}','Aebs.vision_sls.refine_sbc_actor',[
                '--artifact-dir',actor_base,'--barrier-dir',barrier_base,
                '--perception-contract',actor_base/'perception_contract.json','--output-dir',actor_dir,
                '--rounds','8','--steps-per-round','100','--line-search'])
            # This is a development rebind only. Never transfer statistical validity.
            c=json.loads((actor_dir/'perception_contract.json').read_text())
            c.update(independent_trajectory_calibration=False,statistical_status='development tube only')
            c['actor_rebinding_note']='Not a trajectory CP contract; final changed-actor calibration is separate.'
            write(actor_dir/'perception_contract.json',c)
            bdir=out/f'barrier_{k:02d}'
            args=['--artifact-dir',actor_dir,'--perception-contract',actor_dir/'perception_contract.json',
                  '--output-dir',bdir,'--epochs','4000','--seed','41','--nonnegative-activation','relu',
                  '--counterexample-feedback','--raw-region-loss','--drift-loss','linear','--learning-rate','.003' if resume is None else '.0003']
            if resume:
                args+=['--resume',resume]
                feedback=out/f'development_audit_{k-1:02d}/counterexamples.npz'
                if feedback.exists():args+=['--extra-states',feedback]
            run(f'barrier_{k:02d}','Aebs.vision_sls.train_sbc_cp',args)
            audit_dir=out/f'development_audit_{k:02d}'
            run(f'development_audit_{k:02d}','Aebs.vision_sls.verify_sbc_cells',[
                '--artifact-dir',actor_dir,'--barrier-dir',bdir,'--perception-contract',actor_dir/'perception_contract.json',
                '--output-dir',audit_dir,'--stop-on-witness','--max-depth','20','--max-cells','1000000','--actor-bound-method','affine'])
            dev=json.loads((audit_dir/'report.json').read_text())
            dev_pass=dev['continuous_state_numerical_interval_pass']
            if dev_pass: break
            actor_base=actor_dir;barrier_base=bdir;resume=bdir/'sbc.pt'
        if not dev_pass:
            write(out/'report.json',dict(status='development_not_certified',audit=dev));return
        package=out/'package';package.mkdir()
        for n in ['policy.pt','observation.pt','observations.npz','splits.npz']:
            shutil.copy2(actor_dir/n,package/n)
        for n in ['sbc.pt','config.json']: shutil.copy2(bdir/n,package/n)
        run('final_cp','Aebs.vision_sls.robust_replay_cp',['--artifact-dir',package,'--output-dir',out/'final_cp'])
        run('final_audit','Aebs.vision_sls.verify_sbc_cells',[
            '--artifact-dir',package,'--barrier-dir',package,'--perception-contract',out/'final_cp/perception_contract.json',
            '--output-dir',out/'final_audit','--max-depth','20','--max-cells','1000000','--actor-bound-method','affine'])
        cp=json.loads((out/'final_cp/report.json').read_text());q=cp['calibration']['radius_m']
        audit=json.loads((out/'final_audit/report.json').read_text());passed=audit['continuous_state_numerical_interval_pass']
        changed=sha256(package/'policy.pt')!=sha256(source/'policy.pt')
    write(out/'report.json',dict(status='passed_numerical' if passed else 'final_not_certified',
        actor_file_changed=changed,fresh_barrier_initialization=True,final_cp_radius=q,
        development_radius=protocol['development_radius'],
        frontend_unchanged=sha256(package/('patch_projection.pt' if a.task=='lane' else 'observation.pt'))==
                           protocol['bindings']['patch_projection.pt' if a.task=='lane' else 'observation.pt'],
        scope='Conditional robust barrier numerical verification; final CP scope stays simulator/frozen replay specific. No outward-rounded IEEE guarantee.'))


if __name__=='__main__':main()
