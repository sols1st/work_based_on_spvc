"""Post-training diagnostic only: no retraining, reselection or CP claims."""
import json,sys
from pathlib import Path
import numpy as np,torch
from Aebs.lane_keeping.nonlinear_observation import load_nonlinear
root=Path(sys.argv[1]);rows=[]
for seed in [202611171,202611172,202611173]:
 p=root/f'train_{seed}/neural';model,_=load_nonlinear(p/'observation.pt');model.double()
 axis=torch.linspace(-1,1,41,dtype=torch.float64);x=torch.cartesian_prod(axis,axis)
 with torch.no_grad():
  design=torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
  fitted=torch.linalg.lstsq(design,model.nominal(x)).solution
  A=fitted[:2].T;b=fitted[2]
  data=np.load(p/'observations.npz');ids=np.flatnonzero(data['split']=='test')
  y=torch.tensor(data['latent'][ids],dtype=torch.float64);truth=torch.tensor(data['normalized_state'][ids],dtype=torch.float64)
  estimated=torch.linalg.solve(A,(y-b).T).T;neural=model.decode(y)
  delta=(estimated-neural).abs();err=estimated-truth
  rows.append(dict(seed=seed,affine_matrix=A.tolist(),affine_bias=b.tolist(),
      test_decoded_mse_with_affine_inverse=float(err.square().mean()),
      test_mae_physical_with_affine_inverse=(err.abs().mean(0)*torch.tensor([.85,.30])).tolist(),
      max_decoded_change_from_neural_inverse=delta.max(0).values.tolist(),
      mean_decoded_change_from_neural_inverse=delta.mean(0).tolist()))
result=dict(scope='Replace each learned h by its best affine fit on a deterministic state grid, retaining its learned visual head. Static diagnostic only, no new CP/certificate. This is not a separately trained affine-control arm.',rows=rows)
(root/'affine_replacement_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
