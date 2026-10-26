"""Direct split CP on raw trajectory residuals; never read the old error bound."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--traces',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--protocol',type=Path,required=True)
args=parser.parse_args()
protocol=json.loads(args.protocol.read_text())
alpha=protocol['alpha']
metadata=json.loads((args.traces.parent/'summary.json').read_text())
assert metadata['random_only'] is True
assert metadata['seed']==protocol['calibration']['seed']
assert metadata['rollouts']==protocol['calibration']['rollouts']
data=np.load(args.traces)
residual=data['residuals']
assert residual.shape==(protocol['calibration']['rollouts'],5,35)
assert np.isfinite(residual).all()
point_scores=np.linalg.norm(residual,axis=1)
scores=point_scores.max(axis=1)
n=len(scores)
k=math.ceil((n+1)*(1-alpha))
assert k<=n,'Finite CP radius is unavailable for the registered alpha and calibration size.'
radius=float(np.partition(scores,k-1)[k-1])
assert radius>0
args.output.mkdir(parents=True,exist_ok=True)
result={
    'method':'Direct trajectory split conformal, unnormalized raw L2 residual',
    'old_envelope_used':False,'alpha':alpha,'target_coverage':1-alpha,
    'calibration_trajectories':n,'observations_per_trajectory':35,'rank_k':k,
    'radius':radius,'F':'radius * I_5',
    'calibration_trajectory_coverage':float(np.mean(scores<=radius+1e-12)),
    'calibration_point_coverage':float(np.mean(point_scores<=radius+1e-12)),
    'score_min':float(scores.min()),'score_max':float(scores.max()),
    'protocol_sha256':hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
    'calibration_traces_sha256':hashlib.sha256(args.traces.read_bytes()).hexdigest(),
    'calibration_seed':protocol['calibration']['seed'],
    'calibration_controller':protocol['calibration']['controller'],
    'scope':'Trajectory coverage for the frozen reference-policy distribution under exchangeability; '
             'replanned-policy coverage is measured separately, not assumed.',
}
np.savez_compressed(args.output/'direct_cp_scores.npz',trajectory_scores=scores,point_scores=point_scores)
(args.output/'direct_cp_calibration.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
