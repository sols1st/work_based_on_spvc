"""Export a neural SBC and audit its invariant-domain certificate artifacts.

This is a reachable-invariant-domain proof for the changed backup-governed
closed loop, NOT verification of the old unrestricted 8D SPVC candidate.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from .core import INITIAL_STATES,GOAL_LOW,GOAL_HIGH
from .planning_feedback import ReferenceBank
from .planned_sbc import PlannedBarrier
from .geometric_neural_sbc import GeometricNeuralSBC,MARGIN,NORMALS,RADII
from .backup_sbc import initial_cover
from .backup_governor import initial_belief,canonical_action
from .interval_backup import add,point,step,safe,sequence_box
from .train_ppo import sha


def audit_episode_archive(path,bank,model,startup):
    metrics=json.loads((path/'metrics.json').read_text());cfg=metrics['config']
    data=np.load(path/'trajectories.npz');rows=[];all_values=[]
    if cfg['startup_model_sha256']!=sha(startup):raise ValueError('startup calibration mismatch')
    if cfg['posterior_pruning']!='none after startup':raise ValueError('proof needs no posterior pruning')
    calibration=json.loads(Path(startup).read_text())
    for k,row in enumerate(metrics['per_scenario']):
        j=row['scenario'];ref=int(data['references'][k]);ids,box=initial_belief(
            data['observations'][k,0],bank.goals[j],calibration,cfg['bound'])
        if row['backup_reference'] is None or not isinstance(row['backup_reference'],int):
            raise ValueError('this archive auditor expects a bank reference backup')
        tail=np.asarray([canonical_action(u) for u in bank.actions[row['backup_reference']]])
        neural_state=torch.tensor(np.concatenate([data['states'][k],data['estimates'][k]],1),dtype=torch.float64)
        with torch.no_grad():
            values=model(neural_state,torch.full((31,),ref),torch.arange(31)).numpy().ravel()
            geometry=model.geometric(neural_state).numpy().ravel()
        valid=True;tails=True;contained=True;goal_tails=True
        for t in range(31):
            archived=data['belief_boxes'][k,t][:,ids,:]
            valid=valid and np.array_equal(archived,np.stack(box)) and safe(box)
            valid=valid and np.array_equal(np.flatnonzero(data['hypothesis_masks'][k,t]),ids)
            x=data['states'][k,t]
            contained=contained and bool(((x>=box[0])&(x<=box[1])).all(1).any())
            end=sequence_box(box,tail[t:])
            tails=tails and end is not None
            if row['goal_backup_certified']:
                index=int(np.flatnonzero(ids==ref)[0])
                goal_tails=goal_tails and end is not None and bool((end[0][index]>=GOAL_LOW[ref]-.001).all() and
                                                                 (end[1][index]<=GOAL_HIGH[ref]+.001).all())
            if t<30:
                u=data['actions'][k,t]
                valid=valid and np.array_equal(canonical_action(u),u)
                box=step(box,u)
        drift=float(np.max(np.diff(values)))
        record=dict(scenario=j,interval_replay_exact=bool(valid),every_backup_tail_safe=bool(tails),
                    goal_tail_checks=bool(goal_tails),physical_containment=bool(contained),
                    initial_sbc=float(values[0]),minimum_sbc=float(values.min()),maximum_drift=drift,
                    maximum_geometric_term=float(geometry.max()),
                    passed=bool(valid and tails and goal_tails and contained and values[0]<=.305+1e-12 and
                                values.min()>=0 and drift<=-.005+1e-12 and geometry.max()<=1e-10))
        rows.append(record);all_values.append(values)
    return dict(source=str(path),source_archive_sha256=sha(path/'trajectories.npz'),
                success_count=metrics['success_count'],episodes=metrics['episodes'],
                collision_count=metrics['collision_count'],out_of_bounds_count=metrics['out_of_bounds_count'],
                all_audits_passed=all(r['passed'] for r in rows),per_scenario=rows),np.asarray(all_values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--learned-sbc',required=True)
    p.add_argument('--startup-model',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--evaluation',action='append',default=[])
    a=p.parse_args();out=Path(a.output_dir);root=Path(a.candidate)
    if out.exists():p.error('new output directory required')
    torch.set_num_threads(1);out.mkdir(parents=True)
    bank=ReferenceBank(root/'references.npz');base=PlannedBarrier(bank)
    base.load_state_dict(torch.load(a.learned_sbc,map_location='cpu',weights_only=False))
    model=GeometricNeuralSBC(base).eval()
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('finite neural parameters required')
    cp=json.loads(Path(a.startup_model).read_text());bound=cp['initial_bound']
    if cp['projection_sha256']!=sha(root/'projection.pt'):raise ValueError('projection mismatch')
    cover,boxes=initial_cover(bound);np.savez_compressed(out/'initial_braking_cover.npz',**boxes)
    reference_rows=[];reference_low=[];reference_high=[]
    for j in range(15):
        box=add(point(INITIAL_STATES[j:j+1]),(-bound,bound));lows=[box[0][0]];highs=[box[1][0]]
        valid=safe(box)
        for u in bank.actions[j]:
            box=step(box,canonical_action(u));valid=valid and safe(box);lows.append(box[0][0]);highs.append(box[1][0])
        goal=bool((box[0][0]>=GOAL_LOW[j]-.001).all() and (box[1][0]<=GOAL_HIGH[j]+.001).all())
        reference_rows.append(dict(scenario=j,whole_box_safe=bool(valid),whole_box_in_goal=goal))
        reference_low.append(lows);reference_high.append(highs)
    np.savez_compressed(out/'reference_cover.npz',low=np.asarray(reference_low),high=np.asarray(reference_high))
    torch.save(model.state_dict(),out/'neural_sbc.pt')
    shutil.copy2(a.startup_model,out/'startup_model.json')
    for filename in ['references.npz','ppo.zip','projection.pt']:shutil.copy2(root/filename,out/filename)
    shutil.copy2(a.learned_sbc,out/'source_learned_sbc.pt')
    source=out/'source_snapshot';source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source/path.name)
    evaluations=[]
    for n,path in enumerate(a.evaluation):
        report,values=audit_episode_archive(Path(path),bank,model,a.startup_model)
        evaluations.append(report);np.savez_compressed(out/('neural_values_%d.npz'%n),values=values)
    summary=dict(certificate_type='neural SBC on a verified augmented reachable invariant domain',
                 representation='fixed ReLU geometric network + bounded trained Tanh/Sigmoid head + linear finite-horizon clock',
                 controller_changed=True,online_backup_governor_required=True,
                 unrestricted_original_8d_sbc_verified=False,analytic_indicator_is_final_certificate=False,
                 initial_sbc_upper=.305,unsafe_sbc_lower=20.,nonterminal_drift_upper=-.005,
                 proof_domain='physical state contained in retained belief; checked safe backup suffix; valid clock/reference; unsafe stopped states',
                 model='registered RK4, dt=0.15, H=30, zero process noise, per-step enclosure allowance 1e-12',
                 safety_definition='original sampled point-center collision and state bounds with original tolerances',
                 initial_cover=cover,reference_cover=reference_rows,
                 startup_alpha=cp['alpha'],startup_cp_scope='marginal over calibration and one future same-group uniform startup draw; not fixed-state or fixed-calibration guarantee',
                 conditional_safety='pathwise safety if the true initial state is retained; future image errors arbitrary',
                 source_learned_sbc_sha256=sha(a.learned_sbc),geometric_margin=MARGIN,
                 normals=NORMALS.tolist(),obstacle_support_radii=RADII.tolist(),evaluations=evaluations)
    summary['interval_and_replay_checks_passed']=bool(cover['passed'] and all(r['whole_box_safe'] and r['whole_box_in_goal'] for r in reference_rows)
                                                   and bool(evaluations) and all(r['all_audits_passed'] for r in evaluations))
    summary['formal_proof_obligations_document']='docs/current/停车场神经SBC_构造证明与实验_20261103.md'
    summary['formal_program_verification_claimed']=False
    (out/'certificate.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest={str(path.relative_to(out)):sha(path) for path in out.rglob('*') if path.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['certificate_type','initial_sbc_upper','unsafe_sbc_lower','nonterminal_drift_upper','interval_and_replay_checks_passed']},indent=2))
    if not summary['interval_and_replay_checks_passed']:raise SystemExit('certificate audit failed')


if __name__=='__main__':main()
