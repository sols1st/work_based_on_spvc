"""Find a positive expected-drift witness for a recorded upstream stochastic candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


def interval(barrier, lo, hi):
    for ix, layer in enumerate(barrier.layers):
        pos = layer.weight.clamp_min(0)
        neg = layer.weight.clamp_max(0)
        lo, hi = (F.linear(lo,pos,layer.bias)+F.linear(hi,neg),
                  F.linear(hi,pos,layer.bias)+F.linear(lo,neg))
        if ix < len(barrier.layers)-1:
            lo,hi = torch.tanh(lo),torch.tanh(hi)
    return F.softplus(lo).flatten(),F.softplus(hi).flatten()


def cells(n):
    edges = torch.linspace(-1.,1.,n+1,dtype=torch.float64)
    def corners(xs):
        return torch.stack(torch.meshgrid(xs,xs,indexing="ij"),-1).reshape(-1,2)
    return corners(edges[:-1]), corners(edges[1:])


@torch.no_grad()
def audit(args):
    upstream = args.upstream.resolve()
    folder = args.folder.resolve()
    out = folder / "drift_audit.json"
    if out.exists():
        raise FileExistsError(out)
    os.chdir(upstream)
    sys.path[:0] = [str(upstream/"auto_LiRPA"),str(upstream)]
    from Aebs.system.env import Aebs
    from Aebs.VT.train import VTLearner
    torch.set_num_threads(3)
    torch.manual_seed(202611172)
    load = torch.load
    def cpu_load(*items, **kw):
        kw.setdefault("map_location","cpu")
        return load(*items, **kw)
    torch.load = cpu_load
    env = Aebs(.05)
    summary = json.loads((folder/"summary.json").read_text())
    record = summary["best"]
    if record is None:
        out.write_text(json.dumps(dict(status="no_scored_candidate"))+"\n")
        return
    loaded = torch.load(folder/"best.pt")
    assert loaded["iteration"] == record["iteration"]
    torch.set_num_threads(3)
    model = VTLearner([2,16,8,1],env,2.,4.,.1,1.,.95)
    model.l_model.load_state_dict(loaded["barrier"])
    model.p_net.load_state_dict(loaded["policy"])
    policy, barrier = model.p_net.double().eval(), model.l_model.double().eval()
    zlow,zhigh = cells(6)
    # Prefer interior predecessors and require all stochastic successors to remain in the model box.
    lo = torch.from_numpy(env.observation_space.low).double()
    hi = torch.from_numpy(env.observation_space.high).double()
    span = torch.as_tensor(env.noise,dtype=torch.float64)
    d,v = torch.meshgrid(torch.linspace(6.7,15.1,91,dtype=torch.float64)/float(env.std1),
                         torch.linspace(.27,2.76,69,dtype=torch.float64), indexing="ij")
    search = torch.stack([d.flatten(),v.flatten()],1)
    samples = []
    for batch in search.split(256):
        action = policy(batch.new_zeros((len(batch),4)),batch)
        center = env.v_next(batch,action).double()
        current = barrier(batch).flatten()
        next_states = center[:,None,:] + ((zlow+zhigh)*.5)*span
        delta = barrier(next_states.reshape(-1,2)).reshape(len(batch),-1).mean(dim=1) - current
        valid = (current < record["lb_unsafe"]) & ((center-span >= lo) & (center+span <= hi)).all(dim=1)
        for s,c,b,score in zip(batch[valid],center[valid],current[valid],delta[valid]):
            samples.append((float(score),s,c,float(b)))
    samples.sort(key=lambda t:-t[0])
    witness = None
    for _, s, center, b in samples[:20]:
        for n in (16,32,64,128,256,512):
            zlow, zhigh = cells(n)
            floors, ceilings = interval(barrier,center+zlow*span,center+zhigh*span)
            expected_lo = float(floors.mean())
            expected_hi = float(ceilings.mean())
            if expected_lo > b+1e-5:
                shifted = (b-record["domain_min"])/(record["ub_init"]-record["domain_min"])
                witness=dict(state_normalized=s.tolist(),state_physical=[float(s[0])*float(env.std1),float(s[1])],
                             nominal_successor=center.tolist(), noise_halfwidth=span.tolist(), pscore=record["score"],
                             current_barrier=b, unsafe_barrier_lower=record["lb_unsafe"],
                             normalized_current=shifted, normalized_threshold=1./(1.-record["score"]),
                             expected_barrier_lower=expected_lo,expected_barrier_upper=expected_hi,
                             expected_drift_lower=expected_lo-b,
                             expected_drift_normalized_lower=(expected_lo-b)/(record["ub_init"]-record["domain_min"]),
                             equal_mass_cells=n*n)
                break
        if witness:
            break
    outcome=dict(status="positive_expected_drift" if witness else "no_witness_in_search",
                 witness=witness,iteration=record["iteration"], checked_on_uniform_noise=True,
                 numerical="float64 interval operations without directed rounding; independent analytical interval propagation",
                 weights_sha256=hashlib.sha256((folder/"best.pt").read_bytes()).hexdigest())
    out.write_text(json.dumps(outcome,indent=2)+"\n")
    print(outcome,flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("upstream",type=Path)
    cli.add_argument("folder",type=Path)
    audit(cli.parse_args())
