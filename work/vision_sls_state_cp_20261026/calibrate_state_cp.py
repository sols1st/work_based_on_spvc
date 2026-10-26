"""Fresh reference-policy trajectory calibration with a fixed new predictor."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from state_cp_features import predict,bound

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve()
protocol=json.loads((root/'protocol.json').read_text())
model=json.loads((root/'results/state_predictor.json').read_text())
path=root/'results/calibration/diagnostic_traces.npz'
metadata=json.loads((path.parent/'summary.json').read_text())
assert metadata['seed']==protocol['calibration']['seed'] and metadata['random_only'] is True
d=np.load(path);res=np.linalg.norm(d['residuals'],axis=1)
n,T=res.shape;assert n==protocol['calibration']['rollouts']
states=d['observation_states'].transpose(0,2,1).reshape(-1,d['states'].shape[1])
g=predict(states,model).reshape(n,T)
scores=(res-g).max(axis=1)
k=math.ceil((n+1)*(1-protocol['alpha']));assert k<=n
offset=float(np.sort(scores)[k-1])
model.update(cp_offset=offset,alpha=protocol['alpha'],rank_k=k,
             calibration_trajectories=n,calibration_seed=metadata['seed'],
             calibration_traces_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
b=bound(states,model).reshape(n,T)
model.update(calibration_score_coverage=float(np.mean(scores<=offset+1e-12)),
             calibration_bound_point_coverage=float(np.mean(res<=b+1e-8)),
             calibration_bound_trajectory_coverage=float(np.mean((res<=b+1e-8).all(axis=1))),
             calibration_bound_min=float(b.min()),calibration_bound_max=float(b.max()))
(root/'results/state_cp_model.json').write_text(json.dumps(model,indent=2)+'\n')
np.savez_compressed(root/'results/state_cp_scores.npz',trajectory_scores=scores,
                    point_scores=res-g,predictor=g,residual_norm=res,bounds=b)
print(json.dumps(model,indent=2),flush=True)
