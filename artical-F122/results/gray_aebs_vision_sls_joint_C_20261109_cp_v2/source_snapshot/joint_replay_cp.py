"""New AEBS latent-frontend replay CP with an explicit nominal-actor bridge.

The observation library contains p(DINO(I)), not estimated distance. Both
the actor adapter and calibration use measured speed and the learned C.
All statistical claims remain conditional on the declared frozen library.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn
from Combined_network.model import CombinedPolicyNetwork
from .model import VisionSLSPolicy
from .joint_observation import load_joint,JointLatentPolicy
from .rollout import rollout
from .conformal import cp_quantile,validate_contract_units
from .run import sha256


def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def load_semantic_policy(artifact):
    obs=torch.load(artifact/'observation.pt',map_location='cpu',weights_only=False)
    saved=torch.load(artifact/'policy.pt',map_location='cpu',weights_only=False)
    if saved['observation_sha256']!=sha256(artifact/'observation.pt'):raise ValueError('base policy binding mismatch')
    dims=[2]+saved['actor_hidden'];layers=[]
    for a,b in zip(dims[:-1],dims[1:]):layers.extend([nn.Linear(a,b),nn.Tanh()])
    model=VisionSLSPolicy(CombinedPolicyNetwork(nn.Sequential(*layers),nn.Linear(dims[-1],1)),obs['distance_scale_m'])
    model.load_state_dict(saved['state_dict']);model.set_state_distance_scale(saved['metadata']['state_distance_scale_m'])
    return model.eval().requires_grad_(False)


def collect(policy,visual_scale,plant_scale,library,n,seed):
    groups=sorted(set(zip(library['weather'].tolist(),library['color'].tolist())))
    rng=np.random.default_rng(seed);initial=rng.uniform([15,2.5],[16,3],(n,2));appearance=rng.integers(len(groups),size=n)
    scores=np.zeros(n);outcomes=np.empty(n,dtype='U32');steps=np.zeros(n,int);trace=[];max_lookup=0.
    C=policy.C.numpy()[0]
    for g,(weather,color) in enumerate(groups):
        ids=np.flatnonzero(appearance==g)
        if not len(ids):continue
        mask=(library['weather']==weather)&(library['color']==color)
        result=rollout(policy,plant_scale,initial[ids],library['distance_m'][mask],library['observation'][mask],record_trajectories=True)
        for frame in result['trajectories']:
            active=ids[np.array(frame['indices'])]
            y=np.array(frame['observation']);d=np.array(frame['distance_m']);v=np.array(frame['speed_mps'])
            error_latent=y-C[0]*d/visual_scale-C[1]*v
            error_m=visual_scale*error_latent/C[0]
            scores[active]=np.maximum(scores[active],abs(error_m))
            trace.append(np.c_[active,d,v,y,error_latent,error_m])
        outcomes[ids]=result['outcomes'];steps[ids]=result['steps'];max_lookup=max(max_lookup,result['lookup_max_error_m'])
    return dict(initial=initial,appearance=appearance,scores=scores,outcomes=outcomes,steps=steps,
        trace=np.concatenate(trace),trace_columns=np.array(['episode','distance_m','speed_mps','latent','latent_error','decoded_error_m'])),max_lookup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint-dir',type=Path,required=True)
    p.add_argument('--base-artifact',type=Path,required=True)
    p.add_argument('--barrier-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=202611090201)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args(argv)
    args.output_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    for name in ['joint_observation.pt','observations.npz']:
        shutil.copyfile(args.joint_dir/name,args.output_dir/name)
    model,saved=load_joint(args.output_dir/'joint_observation.pt')
    S=saved['distance_scale_m'];C=model.matrix().detach()
    base=load_semantic_policy(args.base_artifact)
    if base.distance_scale_m!=S:raise ValueError('visual scale/nominal actor mismatch')
    policy=JointLatentPolicy(base,C).eval().requires_grad_(False)
    plant_scale=json.loads((args.barrier_dir/'config.json').read_text())['barrier_distance_scale_m']
    library=dict(np.load(args.output_dir/'observations.npz',allow_pickle=False))
    source=Path(__file__).resolve().parent
    bindings={str(v):sha256(v) for v in [args.base_artifact/'policy.pt',args.base_artifact/'observation.pt',
        args.barrier_dir/'sbc.pt',args.barrier_dir/'config.json',args.output_dir/'joint_observation.pt',args.output_dir/'observations.npz']}
    protocol=dict(format='aebs_joint_latent_replay_protocol_v1',C=C.tolist(),visual_scale_m=S,plant_scale_m=plant_scale,
        calibration_n=199,holdout_n=100,calibration_seed=args.seed,holdout_seed=args.seed+1,alpha=.05,horizon=400,
        initial='IID uniform [15,16] m x [2.5,3] m/s',appearance='uniform weather/color group, fixed per episode',
        groups=sorted(set(zip(library['weather'].tolist(),library['color'].tolist()))),
        score='max_t |S/C_d * (p(DINO(I_t))-C_d*d_t/S-C_v*v_t)| including nearest-image mismatch',
        controller='pi_new(y,v)=pi_old((y-C_v*v)/C_d,v); no PPO retraining',
        bridge='if e_y = C_d*delta_m/S then decoded observation = d/S+delta_m/S; verifier uses the same semantic actor with physical radius q_m',
        bindings=bindings,source_sha256={name:sha256(source/name) for name in ['joint_replay_cp.py','joint_observation.py','rollout.py','model.py','conformal.py']},
        distribution='frozen image-library nearest replay; NOT fresh CARLA rendering',
        termination='original stopped AEBS events or 400-step time limit; failures retained',
        rules='freeze frontend/controller/library before calibration; freeze q before holdout; no threshold cap')
    write(args.output_dir/'protocol.json',protocol)
    cal,lookup=collect(policy,S,plant_scale,library,199,args.seed)
    np.savez_compressed(args.output_dir/'calibration.npz',**cal)
    cp=cp_quantile(cal['scores'],.05);q=cp['q_hat']
    contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=[1.]*4,envelope_units='m',score_units='m',calibration=cp,
        source_policy_sha256=sha256(args.base_artifact/'policy.pt'),
        source_observation_sha256=sha256(args.output_dir/'joint_observation.pt'),
        observer_kind='joint latent frontend plus explicit measured-speed decoder',
        semantic_verifier_observation_sha256=sha256(args.base_artifact/'observation.pt'),
        independent_trajectory_calibration=True,formal_image_certificate=False,
        statistical_status='trajectory marginal CP under frozen-library law only',protocol_sha256=sha256(args.output_dir/'protocol.json'))
    validate_contract_units(contract);write(args.output_dir/'perception_contract.json',contract)
    digest=sha256(args.output_dir/'perception_contract.json')
    print(json.dumps(dict(stage='threshold_frozen',radius_m=q,rank=cp['rank'])),flush=True)
    test,test_lookup=collect(policy,S,plant_scale,library,100,args.seed+1)
    np.savez_compressed(args.output_dir/'holdout.npz',**test)
    if any(sha256(path)!=h for path,h in bindings.items()) or sha256(args.output_dir/'perception_contract.json')!=digest:
        raise ValueError('frozen model/library/threshold changed')
    if any(sha256(source/name)!=h for name,h in protocol['source_sha256'].items()):raise ValueError('collection code changed')
    def stats(d):
        return dict(n=len(d['scores']),covered=int((d['scores']<=q).sum()),
            outcomes=dict(zip(*[x.tolist() for x in np.unique(d['outcomes'],return_counts=True)])))
    report=dict(protocol=protocol,radius_m=q,rank=cp['rank'],calibration=stats(cal),holdout=stats(test),
        lookup_max_error_m=max(lookup,test_lookup),certificate_status='not yet audited',
        claim='New joint-frontend trajectory CP for the frozen-library law; no inherited old certificate or live-CARLA claim')
    write(args.output_dir/'report.json',report)
    if args.verify:
        subprocess.run([sys.executable,'-m','Aebs.vision_sls.verify_sbc_cells',
            '--artifact-dir',str(args.base_artifact),'--barrier-dir',str(args.barrier_dir),
            '--perception-contract',str(args.output_dir/'perception_contract.json'),
            '--output-dir',str(args.output_dir/'verification'),'--max-depth','40','--max-cells','1000000',
            '--actor-subdivisions','64','--successor-subdivisions','32','--actor-bound-method','affine'],check=True)
        cert=json.loads((args.output_dir/'verification/report.json').read_text())
        report['certificate_status']='passed_numerical' if cert['continuous_state_numerical_interval_pass'] else 'not_passed'
        report['certificate_sha256']=sha256(args.output_dir/'verification/report.json')
        report['numerical_limit']='real-arithmetic coordinate identity plus numerical bounds; no outward-rounded implementation proof'
        write(args.output_dir/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='protocol'}),flush=True)


if __name__=='__main__':main()
