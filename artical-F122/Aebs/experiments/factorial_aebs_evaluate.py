"""Paired closed-loop CP, controller outcomes and identical distance-tube audits."""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from Aebs.vision_sls.conformal import cp_quantile
from Aebs.vision_sls.deploy import VisionSLSController
from Aebs.vision_sls.finalize_robust_aebs import constant_radius,inclusion_evidence
from Aebs.vision_sls.run import sha256
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint
from .factorial_aebs_inputs import BUDGETS,CAL_SEED,MODES,TEST_SEED,TRAIN_SEEDS
from .matched_aebs_library import SCALE,write
from .matched_aebs_replay import audit,collect,episodes,read


def evaluate(args):
    root=args.root
    protocol=read(root/'protocol.json')
    if protocol['bindings'][str(Path(__file__))]!=sha256(Path(__file__)):
        raise ValueError('evaluation source changed after registration')
    package=args.release/'package'
    policy=VisionSLSController(package).policy
    if policy.distance_scale_m!=SCALE or policy.state_distance_scale_m!=SCALE:
        raise ValueError('control parameter units changed')
    parent=read(args.release/'development_certificate.json')
    outer=read(args.release/'development_contract.json')
    if parent['source_contract_sha256']!=sha256(args.release/'development_contract.json'):
        raise ValueError('parent certificate or contract changed')
    inclusion_evidence(parent,outer,outer,package/'policy.pt',package/'sbc.pt',package/'observation.pt')
    actor64=copy.deepcopy(policy).double()
    barrier,_=load_sbc_checkpoint(package/'sbc.pt',read(package/'config.json'))
    barrier=barrier.double().eval().requires_grad_(False)
    src=args.library
    with np.load(src/'predictions.npz') as old:
        table={k:old[k] for k in ['filename','weather','color','distance_m']}
    mapping={n:i for i,n in enumerate(table['filename'])}
    select={f:i for i,f in enumerate(np.load(root/'ids.npy').tolist())}
    source_rows=np.array([select[f] for f in table['filename']])
    libs={k:np.array([mapping[f] for f in files]) for k,files in read(src/'libraries.json').items()}
    out=root/'replay'
    out.mkdir(exist_ok=False)
    prespec=sha256(root/'protocol.json')
    manifest={}; predictions={}
    for n in BUDGETS:
        for m in ['gray32_mlp',*MODES]:
            for seed in TRAIN_SEEDS:
                key=f'{m}_{n}_{seed}'
                model=root/'models'/key; fit=read(model/'fit.json')
                if fit['protocol_sha256']!=prespec or sha256(model/'prediction.npy')!=fit['prediction']:
                    raise ValueError('model or prediction changed')
                predictions[key]=np.load(model/'prediction.npy')[source_rows]
                manifest[key]=fit
    write(out/'frozen.json',dict(models={k:v['model'] for k,v in manifest.items()},
           protocol=prespec,actor=sha256(package/'policy.pt'),barrier=sha256(package/'sbc.pt'),
           library=sha256(src/'manifest.json'),parent=sha256(args.release/'development_certificate.json')))
    rows={}
    torch.set_num_threads(2)
    for phase,seed,num in [('calibration',CAL_SEED,199),('holdout',TEST_SEED,100)]:
        initials,appearances=episodes(num,seed)
        for key,pred in predictions.items():
            fit=manifest[key];n=fit['budget']
            for lib in ['common',f'strict_{n}']:
                name=f'{lib}__{key}'
                folder=out/name
                folder.mkdir(exist_ok=True)
                ids=libs[lib]
                data=collect(policy,table,ids,pred,initials,appearances)
                np.savez_compressed(folder/(phase+'.npz'),**data)
                if phase=='calibration':
                    cp=cp_quantile(data['scores'],.05)
                    contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=[1.]*4,envelope_units='m',
                                  score_units='m',calibration=cp,coverage={},source_policy_sha256=sha256(package/'policy.pt'),
                                  source_observation_sha256=fit['model'],protocol_sha256=prespec,library=lib,
                                  observations_npz=fit['prediction'],trajectory_sha256=sha256(folder/'calibration.npz'))
                    write(folder/'contract.json',contract)
                    rows[name]=dict(library=lib,budget=n,seed=fit['seed'],frontend=fit['mode'],frames=len(ids),
                        train=fit['n_train'],development=fit['n_dev'],internal_test=fit['test'],steps=fit['gradients'],
                        calibration_queries=int(data['steps'].sum()),q=cp['q_hat'],contract=sha256(folder/'contract.json'),
                        frame_mae=float(np.mean(abs(pred[ids].astype(float)*SCALE-table['distance_m'][ids]))))
                else:
                    row=rows[name]
                    if sha256(folder/'contract.json')!=row['contract']:
                        raise ValueError('threshold changed')
                    statuses=data['outcomes']
                    row.update(holdout_queries=int(data['steps'].sum()),covered=int((data['scores']<=row['q']).sum()),
                       in_fixed_tube=int((data['scores']<=constant_radius(outer)).sum()),
                       mean_peak=float(data['scores'].mean()), max_lookup=float(data['max_lookup_error_m'].max()),
                       success=int((statuses=='success').sum()),unsafe=int((statuses=='unsafe').sum()),
                       early_stop=int((statuses=='stopped_safe_outside_goal').sum()))
        print('completed',phase,flush=True)
    for key,row in rows.items():
        folder=out/key
        result=audit(folder,package,read(folder/'contract.json'),actor64,barrier,parent,outer)
        write(folder/'audit.json',result)
        row['audit']=result['status']
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=next(iter(rows.values())).keys())
        writer.writeheader();writer.writerows(rows.values())
    write(out/'summary.json',rows)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('--release',type=Path,default=Path('results/aebs_robust_synthesis_20261117_release'))
    cli.add_argument('--library',type=Path,default=Path('results/aebs_matched_replay_20261117/library'))
    evaluate(cli.parse_args())
