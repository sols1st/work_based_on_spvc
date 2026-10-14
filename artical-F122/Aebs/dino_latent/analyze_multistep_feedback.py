"""Read-only analysis of stored on-policy trajectories; no new experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from Aebs.dino_latent.diagnose_environment_barrier import write_json
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.train_image_transition_nbc import SignedNBC
from Aebs.dino_latent.nbc_multistep import check_transition_chain,multistep_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error("new analysis output required")
    torch.set_num_threads(1)
    run=json.loads((args.run_dir/"metrics.json").read_text())
    cp=torch.load(args.run_dir/"joint_nbc.pt",map_location="cpu")
    scale=cp["vt_distance_scale_m"]
    model=SignedNBC(scale,cp["hidden"]).eval()
    report=dict(scope="stored training-replay trajectory analysis; not independent validation or a certificate",
        training_run=False,test_used=False,source_metrics_sha256=sha256(args.run_dir/"metrics.json"),
        barrier_steps=5,barrier_lambda=.1,epsilon=.01,gamma=.99,rollouts=[])
    for path in sorted((args.run_dir/"rollouts").glob("*/trajectory.npz")):
        model.load_state_dict(torch.load(path.parent/"scoring_nbc.pt",map_location="cpu"),strict=True)
        with np.load(path) as z:
            x,y=z["physical_x"],z["physical_y"]
            ids,term,trunc=z["episode_ids"],z["terminated"],z["truncated"]
            check_transition_chain(x,y,ids,term,trunc)
            with torch.no_grad():
                c,cn=model(torch.from_numpy(x)),model(torch.from_numpy(y))
                single,_,_=multistep_scores(c,cn,ids,term,trunc,steps=1)
                multi,horizons,counts=multistep_scores(c,cn,ids,term,trunc,steps=5)
            a,b=single.numpy(),multi.numpy()
            unsafe=np.flatnonzero(term & (y[:,0]*scale<=6.) & (y[:,1]>.5))
            warning=dict(eligible_event_lead_pairs=0,single_positive=0,five_step_positive=0,new_positive=0)
            for end in unsafe:
                for lead in range(1,5):
                    start=end-lead
                    if start<0 or ids[start]!=ids[end]: continue
                    warning["eligible_event_lead_pairs"]+=1
                    warning["single_positive"]+=int(a[start]>0)
                    warning["five_step_positive"]+=int(b[start]>0)
                    warning["new_positive"]+=int(a[start]<=0 and b[start]>0)
            report["rollouts"].append(dict(rollout=path.parent.name,steps=len(x),
                data_sha256=sha256(path),scoring_nbc_sha256=sha256(path.parent/"scoring_nbc.pt"),
                valid_windows_per_horizon=counts,full_five_step_starts=int((horizons.numpy()==5).sum()),
                partial_starts=int((horizons.numpy()<5).sum()),single_positive=int((a>0).sum()),
                five_step_positive=int((b>0).sum()),new_positive=int(((a<=0)&(b>0)).sum()),
                unsafe_terminal_events=len(unsafe),pre_unsafe_warning=warning))
    report["total"]={k:sum(r[k] for r in report["rollouts"]) for k in
        ("steps","single_positive","five_step_positive","new_positive","unsafe_terminal_events")}
    report["pre_unsafe_warning"]={k:sum(r["pre_unsafe_warning"][k] for r in report["rollouts"])
        for k in ("eligible_event_lead_pairs","single_positive","five_step_positive","new_positive")}
    write_json(args.output,report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__": main()
