"""Independent randomized enclosure checks for new comparison adapters.

Samples test implementation consistency, not universal interval soundness.
The closed-loop model and actor are independently reevaluated at interior and
corner points, including small/large boxes and actuator saturation regions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .tasks import NAMES, load_task


def check(task, count=80, points=96, seed=2026111801):
    rng = np.random.default_rng(seed)
    worst_step = worst_barrier = 0.0
    failures = []
    safe_claims = 0
    tic = time.monotonic()
    for index in range(count):
        domain_l, domain_h = ((task.initial_lo, task.initial_hi) if index % 2
                              else (task.domain_lo, task.domain_hi))
        center = rng.uniform(domain_l, domain_h)
        radius = (domain_h-domain_l)*rng.uniform(.0001, .25, task.state_dim)
        lo, hi = np.maximum(domain_l, center-radius), np.minimum(domain_h, center+radius)
        if index == 0:
            lo, hi = task.initial_lo.copy(), task.initial_hi.copy()
        if index == 1:
            lo, hi = task.domain_lo.copy(), task.domain_hi.copy()
        x = rng.uniform(lo, hi, (points, task.state_dim))
        # Include every extreme separately on modest dimensions, and random
        # mixed corners rather than only the all-low/all-high diagonal.
        corners = min(points//2, 2**task.state_dim)
        x[:corners] = np.where(rng.integers(2,size=(corners,task.state_dim)),hi,lo)
        e = rng.uniform(task.error_lo,task.error_hi,(points,task.error_dim))
        e[:corners] = np.where(rng.integers(2,size=(corners,task.error_dim)),task.error_hi,task.error_lo)
        w = rng.uniform(task.noise_lo,task.noise_hi,(points,len(task.noise_lo)))
        if len(task.noise_lo):
            w[:corners] = np.where(rng.integers(2,size=(corners,len(task.noise_lo))),task.noise_hi,task.noise_lo)
        context = task.contexts[int(rng.integers(len(task.contexts)))]
        t = int(rng.integers(task.horizon))
        lower,upper = task.interval_step(lo,hi,context=context,t=t)
        nxt = task.step(x,e,w,context=context,t=t)
        violation = float(max(np.max(lower-nxt),np.max(nxt-upper),0))
        worst_step=max(worst_step,violation)
        bl,bu=task.barrier_bounds(lo,hi,t=t)
        bp=np.asarray(task.barrier(x,t=t))
        bviolation=float(max(np.max(bl-bp),np.max(bp-bu),0))
        worst_barrier=max(worst_barrier,bviolation)
        fail = dict(index=index,context=context,step_violation=violation,barrier_violation=bviolation)
        tolerance=1e-9*(1+max(np.max(abs(nxt)),np.max(abs(bp))))
        if violation>tolerance or bviolation>tolerance:
            fail.update(lo=lo.tolist(),hi=hi.tolist(),t=t)
            failures.append(fail)
        if task.classify_safe(lo,hi,t=t)=='safe':
            safe_claims+=1
            if any(task.classify_safe(point,t=t)!='safe' for point in x):
                failures.append(dict(index=index,error='safe_box_contained_non_safe_point'))
    return dict(task=task.name,passed=not failures,boxes=count,points_per_box=points,
                actual_step_samples=count*points,max_step_enclosure_violation=worst_step,
                max_barrier_enclosure_violation=worst_barrier,safe_claims=safe_claims,
                failures=failures,seconds=time.monotonic()-tic,
                metadata=task.metadata,scope='randomized consistency checks, not soundness proof')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',nargs='+',default=list(NAMES),choices=NAMES)
    p.add_argument('--root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--boxes',type=int,default=80)
    args=p.parse_args()
    torch.set_num_threads(1)
    rows=[]
    for name in args.tasks:
        try:
            row=check(load_task(name,args.root),count=args.boxes)
        except Exception as exc:
            import traceback
            row=dict(task=name,passed=False,error=repr(exc),traceback=traceback.format_exc())
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('metadata','failures')},default=str),flush=True)
    report=dict(checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=rows,
                passed=all(r['passed'] for r in rows))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n')
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
