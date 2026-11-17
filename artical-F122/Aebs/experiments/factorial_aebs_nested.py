"""Nested target image budgets with jointly stratified selection within the original 24k split."""
import argparse
import copy
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from Aebs.vision_sls.conformal import cp_quantile
from Aebs.vision_sls.deploy import VisionSLSController
from Aebs.vision_sls.finalize_robust_aebs import inclusion_evidence
from Aebs.vision_sls.run import sha256
from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint
from .factorial_aebs_fit import fit
from .factorial_aebs_inputs import BUDGETS, TRAIN_SEEDS
from .matched_aebs_library import SCALE, write
from .matched_aebs_replay import read,collect,episodes,audit

ALLOCATIONS={400:(240,88,72),10000:(6000,2000,2000),24000:(14172,4908,4920)}
CAL_SEED,TEST_SEED=711845,711846
MODES=['gray32_mlp','gray32']


def nested_order(master,rows,rng):
    groups={}
    for i in rows:
        label=(master['weather'][i],master['color'][i],int((master['distances'][i]-5)//1))
        groups.setdefault(label,[]).append(i)
    all_rows=[];phases=[]
    for group in groups.values():
        group=rng.permutation(group)
        all_rows.extend(group)
        phases.extend((np.arange(len(group))+rng.random())/len(group))
    return np.asarray(all_rows)[np.argsort(phases)]


def register(a):
    a.output.mkdir(parents=True,exist_ok=False)
    master=read(a.budgets/'protocol.json')['24000']
    split=np.array(master['split'])
    seeds={str(s):int(s+712300) for s in TRAIN_SEEDS}
    write(a.output/'protocol.json',dict(main=sha256(a.root/'protocol.json'),master=sha256(a.budgets/'protocol.json'),
        source=sha256(Path(__file__)),fit_source=sha256(Path(__file__).with_name('factorial_aebs_fit.py')),
        training_seeds=TRAIN_SEEDS,allocation=ALLOCATIONS,draw_seeds=seeds,cal=[199,CAL_SEED],held=[100,TEST_SEED],
        groups='within existing split, (weather,car color,one meter distance bin); proportional interleaved random order; exact nested prefixes',
        library='same common 3229 as main; external evaluation bank; nested test images also counted in package',
        primary='same input same training procedure from main; only label budget changes along a nested chain within each seed',
        modes=MODES,library_sha=sha256(a.library/'libraries.json')))
    for s in TRAIN_SEEDS:
        rep=a.output/str(s);rep.mkdir()
        rng=np.random.default_rng(seeds[str(s)])
        orders=[nested_order(master,np.flatnonzero(split==flag),rng) for flag in ('train','validation','test')]
        slices={}
        for n,counts in ALLOCATIONS.items():
            rows=np.sort(np.concatenate([x[:c] for x,c in zip(orders,counts)]))
            assert len(rows)==n and len(np.unique(rows))==n
            slices[str(n)]={k:[master[k][int(i)] for i in rows] for k in ['files','distances','weather','color','split']}
        write(rep/'splits.json',slices)
        write(rep/'protocol.json',dict(root=sha256(a.output/'protocol.json'),split=sha256(rep/'splits.json'),training_seed=s))


def run(a):
    reg=read(a.output/'protocol.json')
    assert reg['source']==sha256(Path(__file__)) and reg['main']==sha256(a.root/'protocol.json')
    assert reg['fit_source']==sha256(Path(__file__).with_name('factorial_aebs_fit.py'))
    jobs=[]
    for s in TRAIN_SEEDS:
        rep=a.output/str(s)
        for name in ['ids.npy','raw_gray.npy','gray32.npy','cache.json']:
            link=rep/name
            if not link.exists():
                link.symlink_to(os.path.relpath(a.root/name,rep))
        b=rep/'budget'
        b.mkdir(exist_ok=True)
        if not (b/'protocol.json').exists():
            (b/'protocol.json').symlink_to('../splits.json')
        jobs.extend((rep,b,m,n,s) for n in [400,10000] for m in MODES)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(fit,jobs))
    torch.set_num_threads(2)
    package=a.release/'package'
    policy=VisionSLSController(package).policy
    if policy.distance_scale_m!=SCALE or policy.state_distance_scale_m!=SCALE:
        raise ValueError('control units')
    outer=read(a.release/'development_contract.json');parent=read(a.release/'development_certificate.json')
    assert parent['source_contract_sha256']==sha256(a.release/'development_contract.json')
    inclusion_evidence(parent,outer,outer,package/'policy.pt',package/'sbc.pt',package/'observation.pt')
    barrier,_=load_sbc_checkpoint(package/'sbc.pt',read(package/'config.json'))
    barrier=barrier.double().eval().requires_grad_(False);actor=copy.deepcopy(policy).double().eval()
    with np.load(a.library/'predictions.npz') as old:
        table={k:old[k] for k in ['filename','weather','color','distance_m']}
    master={f:i for i,f in enumerate(np.load(a.root/'ids.npy').tolist())}
    ids={f:i for i,f in enumerate(table['filename'])}
    common=np.array([ids[f] for f in read(a.library/'libraries.json')['common']])
    rows=[master[f] for f in table['filename']]
    result=[]
    for s in TRAIN_SEEDS:
        rep=a.output/str(s)
        assert read(rep/'protocol.json')['split']==sha256(rep/'splits.json')
        for n in BUDGETS:
            for mode in MODES:
                key=f'{mode}_{n}_{s}'
                folder=(a.root if n==24000 else rep)/'models'/key
                stats=read(folder/'fit.json')
                assert sha256(folder/'prediction.npy')==stats['prediction']
                pred=np.load(folder/'prediction.npy')[rows]
                dest=rep/'replay'/key;dest.mkdir(parents=True,exist_ok=False)
                cal,cc=episodes(199,CAL_SEED);trial,tc=episodes(100,TEST_SEED)
                stream=collect(policy,table,common,pred,cal,cc)
                np.savez_compressed(dest/'calibration.npz',**stream)
                cp=cp_quantile(stream['scores'],.05)
                contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=[1.]*4,envelope_units='m',
                    score_units='m',calibration=cp,coverage={},library='common',source_policy_sha256=sha256(package/'policy.pt'),
                    source_observation_sha256=stats['model'],protocol_sha256=sha256(rep/'protocol.json'),
                    observations_npz=stats['prediction'],trajectory_sha256=sha256(dest/'calibration.npz'))
                write(dest/'contract.json',contract)
                test=collect(policy,table,common,pred,trial,tc)
                np.savez_compressed(dest/'holdout.npz',**test)
                check=audit(dest,package,contract,actor,barrier,parent,outer)
                write(dest/'audit.json',check)
                result.append(dict(seed=s,budget=n,mode=mode,q=cp['q_hat'],covered=int((test['scores']<=cp['q_hat']).sum()),
                    fixed=int((test['scores']<=.45).sum()),success=int((test['outcomes']=='success').sum()),
                    unsafe=int((test['outcomes']=='unsafe').sum()),mean_peak=test['scores'].mean(),audit=check['status'],
                    cal_queries=int(stream['steps'].sum()),held_queries=int(test['steps'].sum())))
    with (a.output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=result[0].keys());w.writeheader();w.writerows(result)
    write(a.output/'summary.json',dict(rows=result,paired_scenarios=[199,100],models=len(result),new_selected_models=len(jobs),
                                       total_active_queries=sum(r['cal_queries']+r['held_queries'] for r in result)))


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('phase',choices=['register','run'])
    cli.add_argument('root',type=Path);cli.add_argument('output',type=Path)
    cli.add_argument('--budgets',type=Path,default=Path('results/aebs_budget_scaling_20261110'))
    cli.add_argument('--library',type=Path,default=Path('results/aebs_matched_replay_20261117/library'))
    cli.add_argument('--release',type=Path,default=Path('results/aebs_robust_synthesis_20261117_release'))
    cli.add_argument('--workers',type=int,default=4)
    args=cli.parse_args()
    (register if args.phase=='register' else run)(args)
