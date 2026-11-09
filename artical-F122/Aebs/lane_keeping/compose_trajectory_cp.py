"""Check a frozen rollout calibration against its actual SBC audit.

Does not turn numerical interval evidence into an IEEE formal proof, nor
marginal CP into fixed-calibration or pass-conditional 95% probability.
"""
import argparse
import json
from pathlib import Path

from .learned_sbc import sha256


def check_bridge(protocol, calibration, certificate):
    q=calibration['radius']
    if not calibration['finite_radius'] or q is None:
        return dict(passed=False,reason='trajectory quantile is infinite')
    if not certificate['passed']:
        return dict(passed=False,reason='SBC audit not passed')
    if certificate['actor_sha256']!=protocol['binding']['actor_sha256']:
        return dict(passed=False,reason='controller differs from calibration controller')
    if certificate['verification']['radius']<q:
        return dict(passed=False,reason='verified radius is smaller than trajectory CP quantile')
    if (not certificate['initial']['passed'] or not certificate['unsafe']['passed']
            or certificate['initial']['bound']>1 or certificate['unsafe']['bound']<20
            or any(not c['passed'] or c['unresolved'] for c in certificate['checks'])):
        return dict(passed=False,reason='inconsistent audit conditions')
    task=certificate['task'];law=protocol['distribution']
    if any(task[k]!=law[k] for k in ['speed','dt','kappas','process_noise']):
        return dict(passed=False,reason='task distribution/model differs from verified model')
    if 'wheelbase' in task and task['wheelbase']!=law['wheelbase']:
        return dict(passed=False,reason='wheelbase mismatch')
    initial=task['initial_halfwidths']
    if any(lo < -h or hi > h for lo,hi,h in zip(law['initial_low'],law['initial_high'],initial)):
        return dict(passed=False,reason='calibration initial support is not certified')
    if certificate.get('source_sha256',{}).get('core.py')!=protocol['binding']['source_sha256']['core.py']:
        return dict(passed=False,reason='plant source binding missing or mismatched')
    if set(c['kappa'] for c in certificate['checks'])!=set(law['kappas']):
        return dict(passed=False,reason='not all sampled curvatures have audits')
    return dict(passed=True,trajectory_radius=q,verified_radius=certificate['verification']['radius'],
                horizon=protocol['horizon'],nominal_marginal_trajectory_coverage=1-calibration['alpha'],
                observer_and_policy_fixed=True,
                safety_implication='All observation errors inside the verified box implies no first exit from the safe set, under the declared numerical model audit.',
                probability_scope='CP marginal probability over fresh calibration sample and new rollout under the fixed registered law. Not a confidence statement for this one fixed calibration set.',
                acceptance_scope='If verification success is data-dependent, the directly justified selection-safe bound is P(audit passes AND new rollout unsafe) <= alpha, assuming sound deterministic verification. Do not assert P(unsafe | audit passes) <= alpha without additional analysis.',
                numerical_scope='The current audit is numerical, not an outward-rounded floating-point execution proof.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cp-dir',type=Path,required=True)
    p.add_argument('--certificate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    protocol=json.loads((args.cp_dir/'protocol.json').read_text())
    calibration=json.loads((args.cp_dir/'calibration.json').read_text())
    certificate=json.loads(args.certificate.read_text())
    if calibration['protocol_sha256']!=sha256(args.cp_dir/'protocol.json'):
        raise ValueError('calibration protocol hash mismatch')
    if calibration['calibration_data_sha256']!=sha256(args.cp_dir/'calibration.npz'):
        raise ValueError('calibration data hash mismatch')
    result=check_bridge(protocol,calibration,certificate)
    result.update(protocol_sha256=sha256(args.cp_dir/'protocol.json'),
                  calibration_report_sha256=sha256(args.cp_dir/'calibration.json'),
                  certificate_sha256=sha256(args.certificate))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
