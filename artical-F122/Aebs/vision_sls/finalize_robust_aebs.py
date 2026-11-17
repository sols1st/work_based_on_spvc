"""Freeze a development-certified actor/barrier, recalibrate, validate inclusion.

A proof for a larger constant observation tube also covers a smaller tube.
No calibration result is used for training. Failed or unknown audits cannot
be transferred. A larger final CP radius requires a new full numerical audit.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from .run import sha256
from .conformal import validate_contract_units


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def constant_radius(contract):
    validate_contract_units(contract)
    if contract['envelope_units']!='m' or not np.array_equal(contract['coefficients'],np.ones(4)):
        raise ValueError('only constant metre-radius contracts supported for inclusion')
    q=contract['calibration']['q_hat']
    if not np.isfinite(q) or q<0:raise ValueError('finite nonnegative radius required')
    return q


def inclusion_evidence(parent,parent_contract,final_contract,policy_path,sbc_path,observer_path):
    if not parent['continuous_state_numerical_interval_pass'] or not parent['region_pass'] or parent['actual_witnesses'] or parent['unresolved_cells']:
        raise ValueError('cannot reuse a failed/unknown parent audit')
    if parent['observation_model']!='tube':raise ValueError('parent must verify the observation tube')
    for c in [parent_contract,final_contract]:
        if c['source_policy_sha256']!=sha256(policy_path) or c['source_observation_sha256']!=sha256(observer_path):
            raise ValueError('changed actor or observer')
    if parent['source_policy_sha256']!=sha256(policy_path) or parent['source_sbc_sha256']!=sha256(sbc_path):
        raise ValueError('changed actor or barrier')
    outer=constant_radius(parent_contract);inner=constant_radius(final_contract)
    if inner>outer:raise ValueError('final uncertainty set is not contained in parent tube')
    return dict(continuous_state_numerical_interval_pass=True,region_pass=parent['region_pass'],
        regions=parent['regions'],actual_witnesses=0,unresolved_cells=0,checked_cells=0,
        parent_checked_cells=parent['checked_cells'],verification_route='constant tube inclusion into already audited superset',
        verified_superset_radius_m=outer,envelope_radius_at_10m=inner,
        source_policy_sha256=sha256(policy_path),source_sbc_sha256=sha256(sbc_path),
        observation_model='tube',full_image_certificate=False,floating_point_formal_proof=False,
        guarantee_scope=parent['guarantee_scope'],arithmetic=parent['arithmetic'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['development-run','actor-dir','barrier-dir','development-audit','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();parent=json.loads((a.development_audit/'report.json').read_text())
    parent_contract=json.loads((a.actor_dir/'perception_contract.json').read_text())
    # Before collecting any final scores, check the complete development evidence.
    inclusion_evidence(parent,parent_contract,parent_contract,a.actor_dir/'policy.pt',a.barrier_dir/'sbc.pt',a.actor_dir/'observation.pt')
    if parent['source_contract_sha256']!=sha256(a.actor_dir/'perception_contract.json'):
        raise ValueError('parent contract identity mismatch')
    a.output.mkdir(parents=True,exist_ok=False);out=a.output
    protocol=json.loads((a.development_run/'protocol.json').read_text())
    protocol.update(selected_actor_dir=str(a.actor_dir),selected_barrier_dir=str(a.barrier_dir),
                    parent_audit=str(a.development_audit),parent_audit_sha256=sha256(a.development_audit/'report.json'),
                    selection='development-only certificate; final CP not consulted',finalizer_sha256=sha256(Path(__file__)))
    write(out/'protocol.json',protocol)
    package=out/'package';package.mkdir()
    for name in ['policy.pt','observation.pt','observations.npz','splits.npz']:shutil.copy2(a.actor_dir/name,package/name)
    for name in ['sbc.pt','config.json']:shutil.copy2(a.barrier_dir/name,package/name)
    shutil.copy2(a.development_audit/'report.json',out/'development_certificate.json')
    shutil.copy2(a.actor_dir/'perception_contract.json',out/'development_contract.json')
    if (a.development_run/'source_snapshot').exists():shutil.copytree(a.development_run/'source_snapshot',out/'source_snapshot')
    if (a.development_run/'source_manifest.json').exists():shutil.copy2(a.development_run/'source_manifest.json',out/'source_manifest.json')
    status=[]
    def run(stage,module,args):
        cmd=[sys.executable,'-m',module,*map(str,args)];status.append(dict(stage=stage,state='running',command=cmd));write(out/'status.json',status)
        with (out/(stage+'.log')).open('w') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        status[-1].update(state='complete' if not r.returncode else 'error',returncode=r.returncode);write(out/'status.json',status)
        r.check_returncode()
    run('final_cp','Aebs.vision_sls.robust_replay_cp',['--artifact-dir',package,'--output-dir',out/'final_cp'])
    c=json.loads((out/'final_cp/perception_contract.json').read_text());q=constant_radius(c)
    if q<=constant_radius(parent_contract):
        result=inclusion_evidence(parent,parent_contract,c,package/'policy.pt',package/'sbc.pt',package/'observation.pt')
        result.update(parent_report_sha256=sha256(out/'development_certificate.json'),source_contract_sha256=sha256(out/'final_cp/perception_contract.json'))
        (out/'final_audit').mkdir();write(out/'final_audit/report.json',result)
        status.append(dict(stage='final_audit',state='complete',method=result['verification_route']));write(out/'status.json',status)
    else:
        run('final_audit','Aebs.vision_sls.verify_sbc_cells',['--artifact-dir',package,'--barrier-dir',package,
            '--perception-contract',out/'final_cp/perception_contract.json','--output-dir',out/'final_audit',
            '--max-depth','20','--max-cells','1000000','--actor-bound-method','affine','--stop-on-witness'])
        result=json.loads((out/'final_audit/report.json').read_text())
    source=Path(protocol['source'])
    write(out/'report.json',dict(status='passed_numerical' if result['continuous_state_numerical_interval_pass'] else 'final_not_certified',
        fresh_barrier_initialization=True,barrier_training_lineage=str(a.development_run),final_cp_radius=q,
        development_radius=constant_radius(parent_contract),frontend_unchanged=sha256(package/'observation.pt')==sha256(source/'observation.pt'),
        scope='Conditional robust barrier, numerical interval evidence, registered frozen replay CP only.'))


if __name__=='__main__':main()
