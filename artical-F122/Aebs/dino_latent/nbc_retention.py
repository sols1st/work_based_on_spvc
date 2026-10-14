"""Train-only retention checks for a signed NBC; NOT a formal verifier.

Old transitions supervise only the barrier, never PPO's on-policy likelihood.
Rejecting a candidate restores both barrier weights and Adam state. This does
not roll back the PPO actor and does not certify the actor/barrier pair.
"""
import copy
import math

import torch
from torch.nn import functional as F
from Aebs.dino_latent.nbc_multistep import multistep_scores


def retention_scores(model, states, successors, initial, unsafe):
    with torch.no_grad():
        c,cn=model(states),model(successors)
        ci,cu=model(initial),model(unsafe)
        return dict(initial_max=float(ci.max()),unsafe_min=float(cu.min()),
            initial_margin_violations=int((ci>-.01+1e-6).sum()),
            unsafe_margin_violations=int((cu<.01-1e-6).sum()),
            initial_sign_violations=int((ci>0).sum()),unsafe_sign_violations=int((cu<=0).sum()),
            old_hinge_mean=float(F.relu(cn-.9*c+.01).mean()),
            old_raw_violations=int(((.9*c-cn)<-1e-6).sum()),
            old_sign_escapes=int(((c<=0)&(cn>0)).sum()))


def retention_failures(candidate, reference):
    """Fixed pretraining reference; never select using validation/test."""
    failures=[]
    if not all(math.isfinite(candidate[k]) for k in ("initial_max","unsafe_min","old_hinge_mean")):
        failures.append("nonfinite_candidate")
    if candidate["initial_margin_violations"]: failures.append("initial_margin")
    if candidate["unsafe_margin_violations"]: failures.append("unsafe_margin")
    if candidate["old_hinge_mean"]>reference["old_hinge_mean"]+1e-6:
        failures.append("old_hinge_mean_increased")
    if candidate["old_raw_violations"]>reference["old_raw_violations"]:
        failures.append("old_raw_violations_increased")
    if candidate["old_sign_escapes"]>reference["old_sign_escapes"]:
        failures.append("old_sign_escapes_increased")
    return failures


def guarded_nbc_update(model, optimizer, x, y, old_x, old_y, train_initial, train_unsafe,
                       guard_x, guard_y, guard_initial, guard_unsafe, reference, generator, steps=10,
                       barrier_steps=1,episode_ids=None,terminated=None,truncated=None):
    weights_before=copy.deepcopy(model.state_dict())
    optimizer_before=copy.deepcopy(optimizer.state_dict())
    model.train()
    loss_value=0.
    for _ in range(steps):
        ids=torch.randint(len(old_x),(2048,),generator=generator)
        if barrier_steps==1:
            current=F.relu(model(y)-.9*model(x)+.01).mean()
        else:
            current=multistep_scores(model(x),model(y),episode_ids,terminated,truncated,steps=barrier_steps)[0].mean()
        replay=F.relu(model(old_y[ids])-.9*model(old_x[ids])+.01).mean()
        init=F.relu(model(train_initial)+.01)
        unsafe=F.relu(.01-model(train_unsafe))
        # Mean terms retain the original label objective; max terms expose
        # rare bad region labels instead of diluting them in a region average.
        loss=.5*(current+replay)+init.mean()+unsafe.mean()+init.max()+unsafe.max()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        loss_value=float(loss.detach())
    model.eval()
    proposed=retention_scores(model,guard_x,guard_y,guard_initial,guard_unsafe)
    failures=retention_failures(proposed,reference)
    if failures:
        model.load_state_dict(weights_before,strict=True)
        optimizer.load_state_dict(optimizer_before)
    kept=retention_scores(model,guard_x,guard_y,guard_initial,guard_unsafe)
    return dict(accepted=not failures,rejection_reasons=failures,proposed=proposed,kept=kept,
                optimized_loss=loss_value,optimizer_steps=steps,
                actor_rollback=False,check_level="finite train-only retention check, NOT continuous safety")
