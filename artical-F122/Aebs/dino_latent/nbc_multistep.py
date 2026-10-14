"""Masked short multi-step NBC hinges, inspired by srlnbc.

Transitions are chronological x_t -> y_t=x_(t+1), with the actual terminal y.
Allow a window to END at termination/truncation; never cross that boundary.
Missing future steps at rollout edges are not fabricated or zero-padded.
Partial windows normalize by their available discounted weights (project
adaptation, rather than the original fixed-denominator implementation).
"""
import numpy as np
import torch


def check_transition_chain(states, successors, episode_ids, terminated, truncated):
    x,y=np.asarray(states),np.asarray(successors)
    ids=np.asarray(episode_ids)
    done=np.asarray(terminated,dtype=bool)|np.asarray(truncated,dtype=bool)
    if (x.shape!=y.shape or x.ndim!=2 or len(x)!=len(ids) or done.shape!=ids.shape or
        not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("invalid chronological transitions")
    same=ids[1:]==ids[:-1]
    if np.any(same & done[:-1]) or np.any((~same) & (~done[:-1])):
        raise ValueError("episode ids and terminal boundaries disagree")
    if same.any() and not np.allclose(y[:-1][same],x[1:][same],rtol=0,atol=1e-6):
        raise ValueError("consecutive states are not recorded successors")


def multistep_scores(current, following, episode_ids, terminated, truncated, steps=5,
                     barrier_lambda=.1, epsilon=.01, gamma=.99):
    """Per-action weighted hinges; differentiable only in supplied C values."""
    if steps<1 or not 0<barrier_lambda<1 or epsilon<=0 or not 0<gamma<=1:
        raise ValueError("invalid multi-step settings")
    c,cn=current.flatten(),following.flatten()
    if c.shape!=cn.shape or not len(c): raise ValueError("C arrays must be nonempty and aligned")
    n=len(c);device=c.device
    ids=torch.as_tensor(episode_ids,device=device).flatten()
    done=(torch.as_tensor(terminated,device=device,dtype=torch.bool).flatten()|
          torch.as_tensor(truncated,device=device,dtype=torch.bool).flatten())
    if ids.shape!=c.shape or done.shape!=c.shape: raise ValueError("boundary arrays differ in length")
    prefix=torch.cat((torch.zeros(1,device=device,dtype=torch.long),done.long().cumsum(0)))
    total=torch.zeros_like(c);denominator=torch.zeros_like(c)
    horizons=torch.zeros(n,device=device,dtype=torch.long)
    counts=[]
    for k in range(1,steps+1):
        length=n-k+1
        if length<=0:
            counts.append(0)
            continue
        start=torch.arange(length,device=device)
        end=start+k-1
        valid=(ids[start]==ids[end]) & ((prefix[end]-prefix[start])==0)
        contraction=(1.-barrier_lambda)**k
        margin=epsilon*(1.-contraction)/barrier_lambda
        hinge=torch.relu(cn[end]-contraction*c[start]+margin)
        weight=gamma**(k-1)
        # F.pad preserves the certificate gradient without detaching windows.
        total=total+torch.nn.functional.pad(hinge*valid*weight,(0,k-1))
        denominator=denominator+torch.nn.functional.pad(valid.to(c.dtype)*weight,(0,k-1))
        horizons=horizons+torch.nn.functional.pad(valid.long(),(0,k-1))
        counts.append(int(valid.sum()))
    return total/denominator.clamp_min(1e-12),horizons,counts
