"""Paired AEBS episode calibration and conditional barrier checks on frozen images."""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from Aebs.vision_sls.conformal import cp_quantile
from Aebs.vision_sls.deploy import VisionSLSController
from Aebs.vision_sls.finalize_robust_aebs import inclusion_evidence,constant_radius
from Aebs.vision_sls.rollout import rollout
from Aebs.vision_sls.run import sha256
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint
from Aebs.vision_sls.verify_sbc_cells import actual_witnesses
from .matched_aebs_library import SCALE,write

CALIBRATION_SEED=202611171901
HOLDOUT_SEED=202611171902


def read(path):
    return json.loads(path.read_text())


def episodes(n,seed):
    rng=np.random.default_rng(seed)
    return rng.uniform([15.,2.5],[16.,3.],(n,2)),rng.integers(0,12,size=n)


def collect(policy,table,ids,observation,initial,appearances):
    groups=sorted(set(zip(table['weather'],table['color'])))
    if len(groups)!=12:
        raise ValueError('expected all twelve appearance groups')
    n=len(initial)
    scores=np.zeros(n);lookup=np.zeros(n);steps=np.zeros(n,int)
    outcomes=np.empty(n,dtype='U32');terminal=np.empty_like(initial)
    mean_abs_error=np.zeros(n); group_counts=[]
    for group,(weather,color) in enumerate(groups):
        subset=ids[(table['weather'][ids]==weather)&(table['color'][ids]==color)]
        group_counts.append(len(subset))
        if not len(subset):
            raise ValueError(f'no test image in {weather} {color}')
        ep=np.flatnonzero(appearances==group)
        if not len(ep):
            continue
        d=table['distance_m'][subset]
        observed=observation[subset]
        def monitor(indices,x,nxt,action,y,end):
            active=ep[indices]
            error=np.abs(y.astype(float)*SCALE-x[:,0])
            scores[active]=np.maximum(scores[active],error)
            mean_abs_error[active]+=error
            mismatch=np.abs(x[:,0,None]-d[None,:]).min(axis=1)
            lookup[active]=np.maximum(lookup[active],mismatch)
        result=rollout(policy,SCALE,initial[ep],d,observed,monitor=monitor)
        outcomes[ep]=result['outcomes'];steps[ep]=result['steps'];terminal[ep]=result['terminal_states']
    if not np.isfinite(scores).all() or not (steps>0).all():
        raise ValueError('invalid episode')
    return dict(initial=initial,appearance=appearances,scores=scores,outcomes=outcomes,steps=steps,
                terminal_states=terminal,episode_mean_abs_error_m=mean_abs_error/steps,
                max_lookup_error_m=lookup,group_counts=np.asarray(group_counts))


def find_witness(policy,barrier,contract):
    # Coarse full-domain search and a denser search near the braking boundary.
    for d0,d1,nd,v0,v1,nv in [(6.00001,16.,160,1e-5,3.,120),(6.00001,7.5,257,1e-5,.8,257)]:
        d,v=torch.meshgrid(torch.linspace(d0,d1,nd,dtype=torch.float64),
                           torch.linspace(v0,v1,nv,dtype=torch.float64),indexing='ij')
        points=torch.stack((d.flatten(),v.flatten()),dim=1)
        for x in points.split(1024):
            bad,drift,y=actual_witnesses(policy,barrier,SCALE,contract,x)
            if bad.any():
                j=int(torch.nonzero(bad)[0,0])
                return dict(state=x[j].tolist(),distance_error_m=float(y[j]*SCALE-x[j,0]),
                            drift=float(drift[j]),observed_normalized=float(y[j]))
    return None


def audit(root,package,contract,actor64,barrier,parent,parent_contract):
    q=constant_radius(contract);outer=constant_radius(parent_contract)
    if q <= outer:
        # The parent audit's input is ANY estimated distance d+e, not images.
        # Its region, actor, unit and full-tube identities were verified in main.
        return dict(status='pass',method='subset of universal distance-error input tube',
                    q=q,outer_q=outer,parent_cells=parent['checked_cells'],
                    regions=parent['regions'],witness=None)
    witness=find_witness(actor64,barrier,contract)
    if witness:
        return dict(status='witness',method='direct pointwise positive drift for registered B',
                    q=q,witness=witness)
    cmd=[sys.executable,'-m','Aebs.vision_sls.verify_sbc_cells','--artifact-dir',str(package),
         '--barrier-dir',str(package),'--perception-contract',str(root/'contract.json'),
         '--output-dir',str(root/'interval'),'--actor-bound-method','affine',
         '--max-depth','20','--max-cells','1000000','--stop-on-witness']
    with (root/'interval.log').open('w') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    rep=read(root/'interval/report.json')
    status='pass' if rep['continuous_state_numerical_interval_pass'] else ('witness' if rep['actual_witnesses'] else 'unresolved')
    return dict(status=status,method='direct numerical interval audit',q=q,report='interval/report.json')


def main(a):
    a.out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    release=a.release;package=release/'package'
    parent=read(release/'development_certificate.json');outer=read(release/'development_contract.json')
    if parent['source_contract_sha256']!=sha256(release/'development_contract.json'):
        raise ValueError('development contract binding mismatch')
    inclusion_evidence(parent,outer,outer,package/'policy.pt',package/'sbc.pt',package/'observation.pt')
    bundle=read(a.library/'manifest.json')
    for f,h in bundle['outputs'].items():
        if sha256(a.library/f)!=h:raise ValueError(f'changed library file {f}')
    with np.load(a.library/'predictions.npz') as z:table=dict(z)
    file_index={f:i for i,f in enumerate(table['filename'])}
    libs=read(a.library/'libraries.json')
    observer_hash={k:v['sha256'] for k,v in bundle['inputs'].items() if k.startswith(('gray_','dino_'))}
    if a.gray:
        gray_manifest=read(a.gray/'manifest.json')
        if gray_manifest['inputs']['raw_library']!=sha256(a.library/'manifest.json') or gray_manifest['predictions']!=sha256(a.gray/'predictions.npz'):
            raise ValueError('gray model library binding mismatch')
        with np.load(a.gray/'predictions.npz') as z:
            if not np.array_equal(z['filename'],table['filename']):raise ValueError('gray predictions reordered')
            table['dino_gray_400']=z['observation'].copy()
        observer_hash={'dino_gray_400':gray_manifest['checkpoint']}
    arms={}
    for variant in observer_hash:
        budget=variant.split('_')[-1]
        for lib in ('strict_'+budget,'common'):
            arms[lib+'__'+variant]=(lib,variant)
    policy=VisionSLSController(package).policy
    if policy.distance_scale_m!=SCALE or policy.state_distance_scale_m!=SCALE:
        raise ValueError('unmatched distance units')
    barrier,_=load_sbc_checkpoint(package/'sbc.pt',read(package/'config.json'))
    barrier=barrier.double().eval().requires_grad_(False);actor64=copy.deepcopy(policy).double()
    recorded=dict(method='fixed shared actor and robust barrier; separate regenerated latent input lookup for each frontend',
          sources={f:sha256(Path(f)) for f in [str(Path(__file__)),
          'Aebs/vision_sls/rollout.py','Aebs/vision_sls/verify_sbc_cells.py','Aebs/system/outcomes.py']},
          cal_seed=CALIBRATION_SEED,hold_seed=HOLDOUT_SEED,calibration_n=199,holdout_n=100,alpha=.05,horizon=400,
          scenario='IID uniform [15,16]x[2.5,3]; IID one of twelve fixed weather/color conditions per episode; no discarded episodes',
          score='episode max absolute prediction in meters minus CURRENT true distance including nearest-image mismatch',
          native_semantics='frozen 32x32 gray real-image SPVC perception head compared with frozen pretrained DINO + projection. No generator here.',
          streams='all arms share initial states and appearances; each evolves according to its own actions',
          release=str(release),actor=sha256(package/'policy.pt'),barrier=sha256(package/'sbc.pt'),
          parent_audit=sha256(release/'development_certificate.json'),library=sha256(a.library/'manifest.json'),
          observer_sha256=observer_hash,strict='held images from individual total budget; exclude train/dev gray pixel collisions',
          common='evaluation-only library removes train/dev of all budgets; extra evaluation images for smaller budgets',
          failure_rule='stop on positive drift witness; otherwise nested universal-tube result or identical direct interval budget',
          arms=arms)
    write(a.out/'protocol.json',recorded)
    pref=sha256(a.out/'protocol.json')
    summary={}
    for phase,(n,seed) in [('calibration',(199,CALIBRATION_SEED)),('holdout',(100,HOLDOUT_SEED))]:
        initial,appearance=episodes(n,seed)
        for label,(lib,variant) in arms.items():
            root=a.out/label;root.mkdir(exist_ok=True)
            ids=np.array([file_index[f] for f in libs[lib]])
            data=collect(policy,table,ids,table[variant],initial,appearance)
            np.savez_compressed(root/(phase+'.npz'),**data)
            if phase=='calibration':
                cp=cp_quantile(data['scores'],.05)
                contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=[1.]*4,envelope_units='m',
                         score_units='m',calibration=cp,coverage={},source_policy_sha256=recorded['actor'],
                         source_observation_sha256=observer_hash[variant],protocol_sha256=pref,library=lib,
                         observations_npz=sha256(a.library/'predictions.npz') if not a.gray else sha256(a.gray/'predictions.npz'),
                         calibration_npz=sha256(root/'calibration.npz'))
                write(root/'contract.json',contract)
                summary[label]=dict(q=cp['q_hat'],rank=cp['rank'],calibration_n=n,calibration_lookup_max=float(data['max_lookup_error_m'].max()),
                        library_images=len(ids),group_counts=data['group_counts'].tolist(),contract_sha256=sha256(root/'contract.json'))
            else:
                row=summary[label]
                counts=dict(zip(*[x.tolist() for x in np.unique(data['outcomes'],return_counts=True)]))
                row.update(holdout_n=n,covered=int((data['scores']<=row['q']).sum()),outcomes=counts,
                         mean_trajectory_max_error_m=float(data['scores'].mean()),
                         mean_episode_mae_m=float(data['episode_mean_abs_error_m'].mean()),
                         holdout_lookup_max=float(data['max_lookup_error_m'].max()),holdout_sha256=sha256(root/'holdout.npz'))
            print(phase,label,summary[label],flush=True)
    for label in arms:
        root=a.out/label;row=summary[label]
        if sha256(root/'contract.json')!=row['contract_sha256']:raise ValueError('changed threshold')
        result=audit(root,package,read(root/'contract.json'),actor64,barrier,parent,outer)
        result.update(sbc_sha256=recorded['barrier'],actor_sha256=recorded['actor'],parent_sha256=recorded['parent_audit'],
                      semantic_observer_sha256=read(root/'contract.json')['source_observation_sha256'],
                      contract_sha256=row['contract_sha256'])
        write(root/'audit.json',result)
        row['audit']=result['status'];row['audit_sha256']=sha256(root/'audit.json')
        print('audit',label,result['status'],flush=True)
    write(a.out/'summary.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('library',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--gray',type=Path)
    parser.add_argument('--release',type=Path,default=Path('results/aebs_robust_synthesis_20261117_release'))
    main(parser.parse_args())
