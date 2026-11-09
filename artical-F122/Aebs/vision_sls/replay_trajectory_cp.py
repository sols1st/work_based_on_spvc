"""Independent episode CP for a declared frozen-library replay distribution.

This does not calibrate future online CARLA images. Appearance is sampled
once per episode and remains fixed. Unsafe/failed episodes are retained.
"""
import argparse,json,copy
from pathlib import Path
import numpy as np
import torch
from .run import sha256
from .deploy import VisionSLSController
from .rollout import rollout
from .conformal import cp_quantile,validate_contract_units


def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def collect(policy,scale,library,n,seed):
    groups=sorted(set(zip(library['weather'].tolist(),library['color'].tolist())))
    rng=np.random.default_rng(seed);initial=rng.uniform([15,2.5],[16,3],(n,2));appearance=rng.integers(len(groups),size=n)
    scores=np.zeros(n);outcomes=np.empty(n,dtype='U32');steps=np.zeros(n,int);max_lookup=0.
    for g,(weather,color) in enumerate(groups):
        ids=np.flatnonzero(appearance==g)
        if not len(ids):continue
        mask=(library['weather']==weather)&(library['color']==color)
        result=rollout(policy,scale,initial[ids],library['distance_m'][mask],library['observation'][mask],record_trajectories=True)
        for frame in result['trajectories']:
            active=ids[np.array(frame['indices'])]
            error=abs(np.array(frame['observation'])*scale-np.array(frame['distance_m']))
            scores[active]=np.maximum(scores[active],error)
        outcomes[ids]=result['outcomes'];steps[ids]=result['steps'];max_lookup=max(max_lookup,result['lookup_max_error_m'])
    return dict(initial=initial,appearance=appearance,scores=scores,outcomes=outcomes,steps=steps),max_lookup


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(2)
    a=args.artifact_dir;library=dict(np.load(a/'observations.npz',allow_pickle=False));scale=json.loads((a/'config.json').read_text())['barrier_distance_scale_m']
    binding={name:sha256(a/name) for name in ['policy.pt','observation.pt','observations.npz']}
    sources={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('rollout.py')]}
    protocol=dict(binding=binding,sources=sources,calibration_n=199,holdout_n=100,calibration_seed=2026110851,holdout_seed=2026110852,
        alpha=.05,horizon=400,score='maximum absolute error in metres over all active control steps, including nearest-image mismatch',
        initial='IID uniform [15,16] x [2.5,3]',appearance='IID uniform over 12 weather/color groups; fixed throughout each episode',
        termination='original success/unsafe/out-of-domain/stop/time-limit, never discard failures',
        distribution='deterministic frozen-image replay, not new online images',conditional_on_frozen_library=True)
    write(args.output_dir/'protocol.json',protocol)
    policy=VisionSLSController(a).policy
    cal,lookup=collect(policy,scale,library,199,2026110851);np.savez_compressed(args.output_dir/'calibration.npz',**cal)
    cp=cp_quantile(cal['scores'],.05);q=cp['q_hat']
    contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=[1.]*4,envelope_units='m',score_units='m',
        calibration=cp,coverage=dict(calibration=dict(samples=199,covered=int((cal['scores']<=q).sum()),mean_radius_m=q)),
        source_policy_sha256=binding['policy.pt'],source_observation_sha256=binding['observation.pt'],
        independent_trajectory_calibration=True,formal_image_certificate=False,
        statistical_status='trajectory-marginal CP for the explicitly frozen replay law only',protocol_sha256=sha256(args.output_dir/'protocol.json'))
    validate_contract_units(contract);write(args.output_dir/'perception_contract.json',contract)
    digest=sha256(args.output_dir/'perception_contract.json')
    print(json.dumps(dict(stage='threshold_frozen',radius_m=q,rank=cp['rank'])),flush=True)
    test,test_lookup=collect(policy,scale,library,100,2026110852);np.savez_compressed(args.output_dir/'holdout.npz',**test)
    if binding!={name:sha256(a/name) for name in binding} or digest!=sha256(args.output_dir/'perception_contract.json'):raise RuntimeError('frozen binding changed')
    report=dict(protocol=protocol,calibration=dict(n=199,rank=cp['rank'],radius_m=q,covered=int((cal['scores']<=q).sum()),
        outcomes=dict(zip(*[x.tolist() for x in np.unique(cal['outcomes'],return_counts=True)]))),
        holdout=dict(n=100,covered=int((test['scores']<=q).sum()),outcomes=dict(zip(*[x.tolist() for x in np.unique(test['outcomes'],return_counts=True)]))),
        lookup_max_error_m=max(lookup,test_lookup),certificate_passed=False,
        claim='Marginal trajectory coverage over calibration and a new IID episode of the registered frozen replay law; not new-image or live CARLA safety.')
    write(args.output_dir/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
