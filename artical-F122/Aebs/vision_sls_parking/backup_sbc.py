"""Analytic nonnegative SBC of the verified backup-governor invariant.

B=0 on sound beliefs admitting their checked backup, B=20 elsewhere.
The proof is inductive and conditional on initialization/model assumptions;
this is not a successfully trained neural SPVC barrier.
"""
import numpy as np
from .core import INITIAL_STATES
from .interval_backup import point,add,safe,backup,step,braking_action


def value(physical_state,governor):
    x=np.asarray(physical_state,dtype=np.float64)
    if x.shape!=(4,) or not np.isfinite(x).all():raise ValueError('finite physical verifier state required')
    lo,hi=governor.box
    member=((x>=lo)&(x<=hi)).all(1).any()
    invariant=bool(member and safe(governor.box) and governor._tail_safe(governor.box,governor.time))
    return 0. if invariant else 20.


def initial_cover(bound=.005,allowance=1e-12):
    """All 15 continuous initial boxes together, not sampled initial points.

    All boxes share the same initial speed interval. The braking policy is
    unchanged under hypothesis deletion, so this also covers every subset.
    """
    box=add(point(INITIAL_STATES),(-bound,bound))
    valid=backup(box,0,allowance)
    lo,hi=box;low=[lo.copy()];high=[hi.copy()];commands=[]
    for _ in range(30):
        u=braking_action((lo,hi));commands.append(u)
        lo,hi=step((lo,hi),u,allowance);low.append(lo.copy());high.append(hi.copy())
    return (dict(passed=bool(valid),initial_box_radius=bound,boxes=15,horizon=30,
                subset_coverage='common speed interval makes braking actions identical after hypothesis deletion',
                outward_interval_arithmetic=True,process_noise='zero',numerical_allowance=allowance),
                dict(low=np.asarray(low),high=np.asarray(high),actions=np.asarray(commands)))


def neural_value(physical_state,governor,barrier):
    """Global hybrid SBC: learned core on the checked invariant, 20 elsewhere.

    Memory (belief, backup and clock) is part of the augmented state. Outside
    the invariant the constant completion makes the drift bound hold for any
    extension of the controller: every successor value is at most 20. This
    completion is essential; the smooth core alone is not globally certified.
    """
    import torch
    if value(physical_state,governor)!=0.:return 20.
    state=np.r_[physical_state,governor.controller.estimate][None]
    with torch.no_grad():
        b=barrier(torch.tensor(state,dtype=torch.float64),
                  torch.tensor([governor.controller.reference]),torch.tensor([governor.time]))
    if not torch.isfinite(b).all():raise ValueError('nonfinite neural SBC value')
    return float(b.item())
