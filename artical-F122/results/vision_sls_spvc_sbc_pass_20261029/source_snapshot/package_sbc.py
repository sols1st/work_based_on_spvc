"""Bundle only matched, numerically interval-verified SBC/actor artifacts."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from .run import sha256


def verified_bundle_status(root):
    """Report saved checks only while every model binding still matches."""
    root=Path(root);summary=json.loads((root/'summary.json').read_text())
    if summary.get('status')!='continuous_state_numerical_interval_checks_passed':
        raise ValueError('bundle does not contain passing SBC verification')
    for name in ['policy.pt','observation.pt','sbc.pt','observations.npz','perception_contract.json']:
        digest=summary['hashes'].get(name)
        if digest is None:raise ValueError('missing verified artifact binding: '+name)
        if sha256(root/name)!=digest:raise ValueError('verified artifact changed: '+name)
    checkpoint=torch.load(root/'sbc.pt',map_location='cpu',weights_only=False)
    metadata=checkpoint['metadata']
    if metadata['policy_sha256']!=sha256(root/'policy.pt') or metadata['observation_sha256']!=sha256(root/'observation.pt'):
        raise ValueError('SBC metadata refers to a different policy or observation model')
    for name in ['tube','replay']:
        report=json.loads((root/'verification'/(name+'.json')).read_text())
        if (not report['continuous_state_numerical_interval_pass'] or report['unresolved_cells'] or report['actual_witnesses'] or
            report['source_policy_sha256']!=metadata['policy_sha256'] or report['source_sbc_sha256']!=metadata['source_raw_sbc_sha256']):
            raise ValueError('SBC verification mismatch: '+name)
    return dict(backend='nonnegative_SBC',numerical_interval_checks_passed=True,
                scope='stopped fixed-image replay; CP tube result conditional on support',
                ieee_floating_point_formal_proof=False,new_image_certificate=False,online_action_filter=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--barrier-dir',type=Path,required=True)
    p.add_argument('--tube-verification',type=Path,required=True)
    p.add_argument('--replay-verification',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output_dir.exists():p.error('choose a new output directory')
    policy_hash=sha256(args.artifact_dir/'policy.pt');barrier_hash=sha256(args.barrier_dir/'sbc.pt')
    reports={name:json.loads(path.read_text()) for name,path in [('tube',args.tube_verification),('replay',args.replay_verification)]}
    for name,r in reports.items():
        if not r['continuous_state_numerical_interval_pass'] or r['actual_witnesses'] or r['unresolved_cells']:
            raise ValueError(name+': verification is incomplete or failed')
        if r.get('observation_model','tube')!=name:
            raise ValueError(name+': wrong observation model in verification report')
        if r['source_policy_sha256']!=policy_hash or r['source_sbc_sha256']!=barrier_hash:
            raise ValueError(name+': verification belongs to different model files')
    cfg=json.loads((args.barrier_dir/'config.json').read_text())
    args.output_dir.mkdir(parents=True);out=args.output_dir
    for name in ['policy.pt','observation.pt','observations.npz','splits.npz','perception_contract.json']:
        shutil.copy2(args.artifact_dir/name,out/name)
    metadata=dict(nonnegative_activation=cfg['nonnegative_activation'],barrier_distance_scale_m=cfg['barrier_distance_scale_m'],
                  policy_sha256=policy_hash,observation_sha256=sha256(out/'observation.pt'),
                  source_raw_sbc_sha256=barrier_hash,reach_probability_target=.95,
                  condition='nonnegative stopped-process supermartingale, B_initial<=1, B_unsafe>=20',
                  strict_legacy_epsilon_point_one_preserved=False,deployment_action_filter=False)
    torch.save(dict(format='vision_sls_nonnegative_sbc_v2',features=[2,16,8,1],metadata=metadata,
                    state_dict=torch.load(args.barrier_dir/'sbc.pt',map_location='cpu',weights_only=True)),out/'sbc.pt')
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    evidence=out/'verification';evidence.mkdir()
    for name,r in reports.items():(evidence/(name+'.json')).write_text(json.dumps(r,indent=2))
    for name in ['expanded_task_check.json','evaluation_reload.json']:
        if (args.artifact_dir/name).exists():shutil.copy2(args.artifact_dir/name,evidence/name)
    summary=dict(status='continuous_state_numerical_interval_checks_passed',backend='SPVC nonnegative SBC adaptation',
                 frontend='frozen DINOv2 + one-dimensional VISION-SLS observation',
                 policy='SBC-refined PPO; no online NCBF or QP',regions=reports['replay']['regions'],
                 reach_probability_target=.95,nonnegative_activation=cfg['nonnegative_activation'],
                 verified_models={name:dict(checked_cells=r['checked_cells'],witnesses=r['actual_witnesses'],unresolved=r['unresolved_cells'],scope=r['guarantee_scope']) for name,r in reports.items()},
                 original_softplus_and_strict_epsilon_configuration_preserved=False,
                 floating_point_formal_proof=False,real_world_image_certificate=False,
                 assumptions=['AEBS stopped at its original terminal events',
                              'tube result requires observation support inside the reference CP tube; marginal CP alone does not prove this',
                              'replay result covers the frozen nearest-image library and has no unmodeled image tail',
                              'float64 numerical interval bounds, without outward rounding or IEEE execution proof'],
                 hashes={name:sha256(out/name) for name in ['policy.pt','observation.pt','sbc.pt','observations.npz','perception_contract.json']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
