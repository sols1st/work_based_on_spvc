"""Retrospective trajectory-wise CP arithmetic; not an independent certification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--outputs',type=Path,required=True)
args=parser.parse_args()
base=Path('/Users/johann/Documents/Codex/2026-10-22/jian/outputs')
settings=[('parking_lot','vision_sls_10server_20261024',8,28),
          ('quadrotor','vision_sls_quad_10server_20261024',20,20)]
args.outputs.mkdir(parents=True,exist_ok=True)
summary={'purpose':'Retrospective feasibility arithmetic on previously inspected data; no fresh CP guarantee.',
         'split_seed':20261024,'alphas':[.1,.05,.01],
         'note':'Exclude the deterministic axis stress rollouts. Split whole random trajectories, never frames.',
         'scenes':{}}
for name,folder,stress,ncal in settings:
    path=base/folder/'visual_rollouts/diagnostic_traces.npz'
    data=np.load(path)
    residual=data['residuals']
    bound=data['error_bounds']
    if bound.ndim==2:
        bound=bound[:,None,:]
    assert np.all(bound>0)
    ratio=np.abs(residual)/np.maximum(bound,1e-12)
    candidates=np.arange(stress,len(residual))
    order=np.random.default_rng(20261024).permutation(candidates)
    cal,test=order[:ncal],order[ncal:]
    info={'input_file':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'total_rollouts':len(residual),'axis_stress_rollouts_excluded_from_calibration':stress,
          'random_rollouts':len(candidates),'calibration_rollouts':cal.tolist(),
          'test_rollouts':test.tolist(),'steps':residual.shape[-1],
          'observation_dimension':residual.shape[1],'scores':{}}
    for score_name,point in [('linf_pdf',ratio.max(axis=1)),
                              ('l2_matches_code',np.linalg.norm(ratio,axis=1))]:
        trajectory=point.max(axis=1)
        evaluations=[]
        for alpha in [.1,.05,.01]:
            k=math.ceil((ncal+1)*(1-alpha))
            q=float(np.partition(trajectory[cal],k-1)[k-1]) if k<=ncal else math.inf
            evaluations.append({'alpha':alpha,'target_coverage':1-alpha,'rank_k':k,
                'scale_q':q if math.isfinite(q) else None,'finite':math.isfinite(q),
                'calibration_trajectory_coverage':float(np.mean(trajectory[cal]<=q+1e-12)),
                'heldout_trajectory_coverage':float(np.mean(trajectory[test]<=q+1e-12)),
                'heldout_trajectories_covered':int(np.sum(trajectory[test]<=q+1e-12)),
                'heldout_trajectory_count':len(test),
                'heldout_point_coverage':float(np.mean(point[test]<=q+1e-12)),
                'axis_stress_trajectory_coverage':float(np.mean(trajectory[:stress]<=q+1e-12))})
        info['scores'][score_name]={
            'original_random_point_coverage':float(np.mean(point[candidates]<=1+1e-12)),
            'original_random_trajectory_coverage':float(np.mean(trajectory[candidates]<=1+1e-12)),
            'random_trajectory_score_min':float(trajectory[candidates].min()),
            'random_trajectory_score_max':float(trajectory[candidates].max()),
            'evaluations':evaluations}
        np.savez_compressed(args.outputs/f'{name}_{score_name}_scores.npz',
                            point_scores=point,trajectory_scores=trajectory,
                            calibration_indices=cal,test_indices=test)
    summary['scenes'][name]=info
(args.outputs/'cp_diagnostic_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
