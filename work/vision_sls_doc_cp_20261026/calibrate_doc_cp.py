"""Independent normalized infinity-residual trajectory CP, with +inf convention."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from doc_cp_features import envelope,positive_scale,box_bound

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();protocol=json.loads((root/'protocol.json').read_text())
meta=json.loads((root/'results/doc_envelope.json').read_text())
path=root/'results/calibration/diagnostic_traces.npz';d=np.load(path)
summary=json.loads((path.parent/'summary.json').read_text())
assert summary['random_only'] and summary['seed']==protocol['calibration']['seed']
n,ny,T=d['residuals'].shape;assert n==protocol['calibration']['rollouts']
states=d['observation_states'].transpose(0,2,1).reshape(-1,d['states'].shape[1])
r=np.max(np.abs(d['residuals']),axis=1);h=positive_scale(states,meta).reshape(n,T)
scores=(r/h).max(axis=1);k=math.ceil((n+1)*(1-protocol['alpha']))
q=float(np.sort(scores)[k-1]) if k<=n else float('inf')
meta.update(cp_multiplier=q,alpha=protocol['alpha'],calibration_rank=k,calibration_trajectories=n,
 calibration_seed=summary['seed'],calibration_traces_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
 protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest())
b=box_bound(states,meta).reshape(n,T)
meta.update(calibration_point_coverage=float((r<=b+1e-8).mean()),
 calibration_trajectory_coverage=float((r<=b+1e-8).all(axis=1).mean()),
 calibration_box_bound_min=float(b.min()),calibration_box_bound_max=float(b.max()),
 calibration_positive_scale_min=float(h.min()),calibration_positive_scale_max=float(h.max()))
(root/'results/doc_cp_model.json').write_text(json.dumps(meta,indent=2)+'\n')
np.savez_compressed(root/'results/doc_cp_scores.npz',trajectory_scores=scores,
 residual_infinity=r,positive_scale=h,box_bounds=b)
print(json.dumps(meta,indent=2),flush=True)
