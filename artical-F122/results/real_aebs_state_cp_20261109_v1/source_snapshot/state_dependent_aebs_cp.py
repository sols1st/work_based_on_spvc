"""State-dependent Bernstein CP for the 2026-11-09 RGB AEBS artifact.

Follows the 2026-10-29 construction: fit b0(d) with nonnegative Bernstein
coefficients on train residuals, then split-CP the trajectory score
max_t |d_hat_t-d_t| / b0(d_t). The envelope returned by b0 is in metres.
"""
import argparse,json,math
from pathlib import Path
import numpy as np, torch
from .conformal import basis,cp_quantile,envelope,fit_scale,validate_contract_units
from .deploy import VisionSLSController
from .rollout import rollout
from .run import sha256

def write(path,obj): path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def collect(policy,scale,library,n,seed,coeff):
    groups=sorted(set(zip(library['weather'].tolist(),library['color'].tolist())))
    rng=np.random.default_rng(seed); initial=rng.uniform([15,2.5],[16,3],(n,2)); appearance=rng.integers(len(groups),size=n)
    score=np.zeros(n); max_b0=np.zeros(n); sum_radius=np.zeros(n); count=np.zeros(n,dtype=int)
    outcomes=np.empty(n,dtype='U32'); steps=np.zeros(n,dtype=int); max_lookup=0.
    for g,(weather,color) in enumerate(groups):
        ids=np.flatnonzero(appearance==g)
        if not len(ids): continue
        mask=(library['weather']==weather)&(library['color']==color)
        result=rollout(policy,scale,initial[ids],library['distance_m'][mask],library['observation'][mask],record_trajectories=True)
        max_lookup=max(max_lookup,result['lookup_max_error_m'])
        for frame in result['trajectories']:
            active=ids[np.array(frame['indices'])]
            d=np.asarray(frame['distance_m'],dtype=float); obs=np.asarray(frame['observation'],dtype=float)*scale
            radius=envelope(d,coeff,1.)
            err=np.abs(obs-d); score[active]=np.maximum(score[active],err/radius)
            max_b0[active]=np.maximum(max_b0[active],radius)
            sum_radius[active]+=radius; count[active]+=1
        outcomes[ids]=result['outcomes']; steps[ids]=result['steps']
    return dict(initial=initial,appearance=appearance,score=score,max_b0=max_b0,
                mean_radius=sum_radius/np.maximum(count,1),count=count,outcomes=outcomes,steps=steps),max_lookup

def main():
    p=argparse.ArgumentParser(); p.add_argument('--artifact-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--degree',type=int,default=3)
    p.add_argument('--fit-samples',type=int,default=5000); p.add_argument('--seed',type=int,default=7)
    args=p.parse_args()
    if args.output_dir.exists(): raise SystemExit('output exists')
    args.output_dir.mkdir(parents=True); out=args.output_dir
    torch.set_num_threads(4)
    a=args.artifact_dir; library=dict(np.load(a/'observations.npz',allow_pickle=False))
    scale=json.loads((a/'config.json').read_text())['barrier_distance_scale_m']
    # fit b0(d) on train static residuals, then split-CP the trajectory score on a disjoint replay set
    lib=dict(np.load(a/'observations.npz',allow_pickle=False)); split=lib['split'].astype(str); tr=np.where(split=='train')[0]
    rng=np.random.default_rng(args.seed); tr=tr[rng.choice(len(tr),min(args.fit_samples,len(tr)),replace=False)]
    residual=np.abs(lib['observation'][tr]*scale-lib['distance_m'][tr])
    coeff,fit=fit_scale(lib['distance_m'][tr],residual,degree=args.degree,slack_penalty=20.,floor=.01)
    write(out/'fit.json',dict(coefficients=coeff.tolist(),fit=fit,samples=len(tr),
                              static_mae_m=float(residual.mean()),static_max_m=float(residual.max())))
    policy=VisionSLSController(a).policy
    cal,lookup=collect(policy,scale,library,199,2026110851,coeff)
    q=cp_quantile(cal['score'],.05)['q_hat']
    # trajectory-level radius statistics
    cal['radius_q']=q*cal['mean_radius']; cal['max_radius_q']=q*cal['max_b0']
    test,lookup2=collect(policy,scale,library,100,2026110852,coeff)
    test['radius_q']=q*test['mean_radius']; test['max_radius_q']=q*test['max_b0']
    cal_cov=int((cal['score']<=q).sum()); test_cov=int((test['score']<=q).sum())
    np.savez_compressed(out/'calibration.npz',**cal); np.savez_compressed(out/'holdout.npz',**test)
    contract=dict(format='vision_sls_state_cp_reference_v1',coefficients=coeff.tolist(),
        envelope_units='m',score_units='dimensionless',fit=fit,
        calibration=dict(q_hat=q,rank=int(math.ceil((199+1)*.95)),n=199,alpha=.05,marginal_target=.95,
                         coverage_lower_at_confidence=float(cp_quantile(cal['score'],.05)['coverage_lower_at_confidence'])),
        coverage=dict(calibration=dict(samples=199,covered=cal_cov,mean_radius_m=float(cal['radius_q'].mean()),
                                       max_radius_m=float(cal['max_radius_q'].max())),
                      holdout=dict(samples=100,covered=test_cov,mean_radius_m=float(test['radius_q'].mean()),
                                   max_radius_m=float(test['max_radius_q'].max()))),
        source_policy_sha256=sha256(a/'policy.pt'),source_observation_sha256=sha256(a/'observation.pt'),
        protocol=dict(calibration_seed=2026110851,holdout_seed=2026110852,horizon=400,
                      initial='IID uniform [15,16]x[2.5,3]',appearance='12 weather/color groups, fixed per episode',
                      distribution='frozen-image nearest replay',state_envelope='b0(d) degree %d nonnegative Bernstein'%args.degree),
        independent_trajectory_calibration=True,formal_image_certificate=False,
        statistical_status='trajectory-marginal CP for the frozen replay law; state-dependent scale b0(d)')
    validate_contract_units(contract); write(out/'perception_contract.json',contract)
    report=dict(calibration=dict(n=199,covered=cal_cov,q=q),holdout=dict(n=100,covered=test_cov),
                fit=fit,radius=dict(cal_mean_m=float(cal['radius_q'].mean()),cal_max_m=float(cal['max_radius_q'].max()),
                                    test_mean_m=float(test['radius_q'].mean()),test_max_m=float(test['max_radius_q'].max())),
                lookup_max_error_m=max(lookup,lookup2),source_policy_sha256=contract['source_policy_sha256'],
                source_observation_sha256=contract['source_observation_sha256'],
                note='state-dependent CP, not yet an SBC certificate')
    write(out/'report.json',report); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
