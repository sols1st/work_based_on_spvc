"""Recompute calibration ranks and diagnose held-out state extrapolation."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from state_cp_features import features, predict, bound

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();out=root/'results'
protocol=json.loads((root/'protocol.json').read_text())
model=json.loads((out/'state_cp_model.json').read_text())
cal=np.load(out/'calibration/diagnostic_traces.npz')
n,_,T=cal['residuals'].shape
states=cal['observation_states'].transpose(0,2,1).reshape(-1,cal['states'].shape[1])
raw=np.linalg.norm(cal['residuals'],axis=1)
scores=(raw-predict(states,model).reshape(n,T)).max(axis=1)
k=math.ceil((n+1)*(1-protocol['alpha']))
q=float(np.sort(scores)[k-1]);assert abs(q-model['cp_offset'])<1e-12
assert hashlib.sha256((out/'calibration/diagnostic_traces.npz').read_bytes()).hexdigest()==model['calibration_traces_sha256']
assert protocol['evaluation']['seed']!=protocol['calibration']['seed']
result={'calibration_rank_recomputed':k,'calibration_offset_recomputed':q,
        'independent_trajectory_scores_equal':bool(np.allclose(scores,np.load(out/'state_cp_scores.npz')['trajectory_scores'])),
        'calibration_and_test_seeds_distinct':True,'policies':{}}
for phase in ['baseline','state_cp']:
    d=np.load(out/phase/'diagnostic_traces.npz');nx=d['states'].shape[1]
    s=d['observation_states'].transpose(0,2,1).reshape(-1,nx)
    g=predict(s,model);b=bound(s,model)
    norms=np.linalg.norm(d['residuals'],axis=1).reshape(-1)
    floor=b<=model['floor']+1e-12
    f=features(s,model['scene'])
    lower=np.array(model['feature_train_min']);upper=np.array(model['feature_train_max'])
    outside=((f<lower-1e-8)|(f>upper+1e-8)).any(axis=1)
    bad=norms>b+1e-8
    result['policies'][phase]={'points':len(s),'floor_points':int(floor.sum()),
       'floor_point_fraction':float(floor.mean()),'uncovered_points':int(bad.sum()),
       'uncovered_floor_points':int((bad&floor).sum()),
       'uncovered_outside_training_coordinate_ranges_points':int((bad&outside).sum()),
       'predictor_min':float(g.min()),'predictor_max':float(g.max()),
       'feature_min':f.min(axis=0).tolist(),'feature_max':f.max(axis=0).tolist(),
       'mean_bound':float(b.mean()),'mean_raw_residual_l2':float(norms.mean()),
       'per_observation_step_coverage':(norms.reshape(50,-1)<=b.reshape(50,-1)+1e-8).mean(axis=0).tolist()}
(out/'calibration_and_extrapolation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
