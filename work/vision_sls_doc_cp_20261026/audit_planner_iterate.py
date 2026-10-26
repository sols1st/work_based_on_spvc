"""Read-only bound diagnostics on the planner's preserved iterates."""
import argparse,json
from pathlib import Path
import numpy as np
from doc_cp_features import envelope,box_bound
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();out=root/'results';file=out/'planner_last_iterate.npz'
if file.exists():
    d=np.load(file);m=json.loads((out/'doc_cp_model.json').read_text());audit={}
    nx=4 if m['scene']=='parking' else 10
    for key in d.files:
        if d[key].shape[0]!=nx:continue
        states=d[key].T;b=envelope(states,m);c=box_bound(states,m)
        audit[key]={'shape':list(d[key].shape),'unclipped_envelope_min':float(b.min()),
          'unclipped_envelope_max':float(b.max()),'box_min':float(c.min()),
          'box_max':float(c.max()),'scale_floor_points':int(np.sum(b<=m['epsilon']))}
    (out/'planner_iterate_bound_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
