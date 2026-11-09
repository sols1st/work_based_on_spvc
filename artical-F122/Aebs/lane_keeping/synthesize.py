"""Reproducible AEBS-style learner/verifier/actor-feedback lane pipeline.

The current policy is retained if a learned certificate already verifies it.
Failed/unknown audits are kept as development evidence, never overwritten.
"""
import argparse
import json
import shutil
from pathlib import Path

from . import train_joint, verify_learned, refine_actor
from .learned_sbc import sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--rounds',type=int,default=4)
    p.add_argument('--epochs',type=int,default=3000)
    p.add_argument('--seed',type=int,default=41)
    p.add_argument('--radius',type=float,default=.08)
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    if args.rounds<1 or args.epochs<1:raise ValueError('positive budgets required')
    args.output_dir.mkdir(parents=True)
    initial_actor=args.actor;actor=args.actor;resume=None;feedback=None;history=[]
    for round_id in range(args.rounds):
        out=args.output_dir/f'round_{round_id:02d}';out.mkdir()
        train_args=['--actor',str(actor),'--output-dir',str(out/'train'),'--epochs',str(args.epochs),
                    '--seed',str(args.seed+round_id),'--radius',str(args.radius)]
        if resume:train_args.extend(['--resume',str(resume),'--learning-rate','.001'])
        if feedback:train_args.extend(['--extra-states',str(feedback)])
        train_joint.main(train_args)
        barrier=out/'train/sbc.pt'
        verify_learned.main(['--actor',str(actor),'--sbc',str(barrier),'--output-dir',str(out/'audit'),
                             '--radius',str(args.radius),'--grid','40','--depth','16',
                             '--error-splits','4','--stop-on-witness'])
        cert=json.loads((out/'audit/certificate.json').read_text())
        entry=dict(round=round_id,actor=str(actor),sbc=str(barrier),verified=cert['passed'],
                   initial_pass=cert['initial']['passed'],unsafe_pass=cert['unsafe']['passed'],
                   unresolved=sum(c['unresolved'] for c in cert['checks']),
                   actual_witnesses=sum(c['actual_witnesses'] for c in cert['checks']))
        history.append(entry)
        (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
        if cert['passed']:
            final=args.output_dir/'verified';final.mkdir()
            shutil.copy2(actor,final/'actor.pt');shutil.copy2(barrier,final/'sbc.pt')
            shutil.copy2(out/'audit/certificate.json',final/'certificate.json')
            report=dict(passed=True,rounds=round_id+1,source_actor_sha256=sha256(initial_actor),
                        final_actor_sha256=sha256(final/'actor.pt'),
                        actor_refinement_triggered=any(h.get('actor_refinement') for h in history),
                        model_class='AEBS NonnegativeSBC [2,16,8,1], Tanh hidden, ReLU output',
                        geometric_teacher=False,scope=cert['scope'])
            (args.output_dir/'report.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return
        feedback=out/'audit/feedback.npz';resume=barrier
        # An invalid region separation is a barrier problem, not evidence
        # that changing the actor is necessary. Repair the SBC first.
        if cert['initial']['passed'] and cert['unsafe']['passed'] and round_id+1<args.rounds:
            refine_actor.main(['--actor',str(actor),'--sbc',str(barrier),
                               '--feedback',str(feedback),'--output-dir',str(out/'actor_feedback'),
                               '--radius',str(args.radius),'--rounds','4'])
            actor=out/'actor_feedback/actor.pt'
            entry['actor_refinement']=str(out/'actor_feedback/report.json')
            (args.output_dir/'history.json').write_text(json.dumps(history,indent=2))
    (args.output_dir/'report.json').write_text(json.dumps(dict(passed=False,rounds=args.rounds,
        reason='budget exhausted; failed and unresolved audits retained',history=history),indent=2))


if __name__=='__main__':main()
