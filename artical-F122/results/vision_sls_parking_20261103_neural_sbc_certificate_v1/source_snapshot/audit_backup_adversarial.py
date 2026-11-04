"""Replay old counterexample error laws and extreme later images with the governor.

Startup uses recorded center images from the independent startup collection.
Later images are synthetic adversarial inputs, not claimed to be DINO outputs.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .core import INITIAL_STATES,nominal_observation,dynamics,constraints
from .planning_feedback import ReferenceBank
from .planned_sbc import PlannedBarrier
from .geometric_neural_sbc import GeometricNeuralSBC
from .backup_governor import BackupGovernor
from .backup_sbc import neural_value
from .perception_error import radius
from .train_migration import load_ppo
from .train_ppo import sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--certificate',required=True)
    p.add_argument('--startup-samples',required=True);p.add_argument('--old-error-model',required=True)
    p.add_argument('--output-dir',required=True);a=p.parse_args();root=Path(a.certificate);out=Path(a.output_dir)
    if out.exists():p.error('new output directory required')
    torch.set_num_threads(1);out.mkdir(parents=True)
    bank=ReferenceBank(root/'references.npz');policy=load_ppo(root/'ppo.zip',device='cpu')
    cp=json.loads((root/'startup_model.json').read_text());old=json.loads(Path(a.old_error_model).read_text())
    model=GeometricNeuralSBC(PlannedBarrier(bank));model.load_state_dict(torch.load(root/'neural_sbc.pt',weights_only=False))
    data=np.load(a.startup_samples);first=data['observations'][data['splits']=='center']
    trials=[(0,'old_negative_corner',scale) for scale in [1.,.5,.25,.1]]+[(j,'extreme_alternating',None) for j in range(15)]
    rows=[];trajectories=[];barriers=[];all_actions=[]
    for j,mode,scale in trials:
        x=INITIAL_STATES[j].copy();g=BackupGovernor(bank,policy,old,initial_jitter=cp['initial_bound'],startup_model=cp)
        g.reset(first[j],bank.goals[j]);xs=[x.copy()];bs=[neural_value(x,g,model)];actions=[];unsafe=False
        for t in range(30):
            captured=x.copy();u=g.action();x=dynamics(x,u);actions.append(u)
            if mode=='old_negative_corner':
                e=np.zeros(3) if t==0 else -scale*radius(captured[None],old)[0]
                y=nominal_observation(captured)+e
            else:y=np.array([np.nan,np.inf,-np.inf]) if t%3==0 else np.full(3,(-1)**t*999999.)
            g.receive(y);xs.append(x.copy());bs.append(neural_value(x,g,model))
            status=constraints(x,j);unsafe=unsafe or status['collision'] or not status['in_bounds']
        drift=float(np.diff(bs).max());ok=bool(not unsafe and min(bs)>=0 and max(bs)<=.305+1e-12 and drift<=-.005+1e-12)
        row=dict(scenario=j,input_type=mode,scale=scale,safe=not unsafe,in_goal=status['in_goal'],
                 maximum_sbc=float(max(bs)),maximum_drift=drift,passed=ok,interventions=g.interventions)
        print(json.dumps(row),flush=True);rows.append(row);trajectories.append(xs);barriers.append(bs);all_actions.append(actions)
    np.savez_compressed(out/'trajectories.npz',states=np.asarray(trajectories),sbc=np.asarray(barriers),actions=np.asarray(all_actions))
    result=dict(trials=rows,all_passed=all(r['passed'] for r in rows),
                source_certificate_sha256=sha(root/'certificate.json'),startup_samples_sha256=sha(a.startup_samples),
                old_error_model_sha256=sha(a.old_error_model),actual_dino_later_images=False,
                probability_estimate=False,scope='sampled regression checks; universal later-error argument is the invariant proof')
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_passed']:raise SystemExit('adversarial regression failed')


if __name__=='__main__':main()
