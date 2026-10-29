"""Discrete-time, structured neural CBF adaptation for frozen AEBS policies.

h(d,v)=d-6-[g(v)-g(.5)]-delta with a monotone convex ReLU g. This guarantees
negative h throughout the original unsafe rectangle. It is NOT the original
nonnegative stochastic barrier, nor a port of the paper's continuous-time QP.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MonotoneNCBF(nn.Module):
    def __init__(self,knots=12,delta=.02):
        super().__init__()
        self.register_buffer('knots',torch.linspace(0.,2.75,knots))
        self.weights=nn.Parameter(torch.full((knots,),-2.5))
        self.base=nn.Parameter(torch.tensor(.5))
        self.delta=float(delta)

    def g(self,v):
        return F.softplus(self.base)*v+(F.relu(v[...,None]-self.knots)*F.softplus(self.weights)).sum(-1)

    def forward(self,state):
        v=state[...,1]
        return state[...,0]-6.-(self.g(v)-self.g(torch.full_like(v,.5)))-self.delta

    def exact_region_bounds(self):
        with torch.no_grad():
            return dict(initial_min=float(self(torch.tensor([[15.,3.]]))),
                        unsafe_max=float(self(torch.tensor([[6.,.5]]))),
                        method='analytic monotonicity, full initial and unsafe rectangles')


def sequential_bounds(net,lower,upper):
    """Differentiable interval arithmetic for the actual frozen PPO actor."""
    for layer in net:
        if isinstance(layer,nn.Linear):
            pos,neg=layer.weight.clamp(min=0),layer.weight.clamp(max=0)
            lower,upper=F.linear(lower,pos,layer.bias)+F.linear(upper,neg,None),F.linear(upper,pos,layer.bias)+F.linear(lower,neg,None)
        elif isinstance(layer,nn.Tanh): lower,upper=lower.tanh(),upper.tanh()
        elif isinstance(layer,nn.ReLU): lower,upper=lower.relu(),upper.relu()
        else: raise TypeError('unsupported interval layer: '+type(layer).__name__)
    return lower,upper


@torch.no_grad()
def action_bounds(policy,states,radius_m,subdivisions=32,batch_size=512):
    """All y in [d-b(d),d+b(d)]/scale; speed is measured, not inferred."""
    if subdivisions<1: raise ValueError('positive subdivisions required')
    states=torch.as_tensor(states,dtype=torch.float32)
    radius=torch.as_tensor(radius_m,dtype=torch.float32).reshape(-1)
    if len(states)!=len(radius) or (radius<0).any() or not torch.isfinite(radius).all():
        raise ValueError('finite nonnegative radii paired with states required')
    actor=nn.Sequential(*policy.controller_net.mlp_extractor,policy.controller_net.action_net)
    output=[]
    for start in range(0,len(states),batch_size):
        s=states[start:start+batch_size]; r=radius[start:start+batch_size]
        endpoints=s[:,0,None]+r[:,None]*torch.linspace(-1.,1.,subdivisions+1)
        lo=torch.stack([endpoints[:,:-1]/policy.distance_scale_m,s[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
        hi=torch.stack([endpoints[:,1:]/policy.distance_scale_m,s[:,1,None].expand(-1,subdivisions)],-1).reshape(-1,2)
        lb,ub=sequential_bounds(actor,lo,hi)
        output.append(torch.stack([lb.reshape(len(s),-1).amin(1).clamp(-3,3),ub.reshape(len(s),-1).amax(1).clamp(-3,3)],1))
    return torch.cat(output)


def successors(states,action_intervals):
    """Exact AEBS successor box. No added artificial distance/speed noise."""
    s=torch.as_tensor(states,dtype=torch.float32); a=torch.as_tensor(action_intervals,dtype=s.dtype)
    d=s[:,0]-.05*s[:,1]
    low=torch.stack([d,(s[:,1]-.05*a[:,1]).clamp(0,3)],1)
    high=torch.stack([d,(s[:,1]-.05*a[:,0]).clamp(0,3)],1)
    return low,high


def raw_constraints(model,initial,unsafe,states,next_high,decay=.05):
    """CP-NCBF q1,q2 and discrete robust q3 (each column has its own draw).

    The system stops at existing safe terminal events. Setting their successor
    value to 32 exceeds max h over the domain; mixed terminal/live boxes retain
    the worst raw bound. Only h>=0 live states require forward invariance.
    """
    h=model(states)
    next_min=model(next_high)  # h increasing in d and decreasing in v
    all_safe_terminal=((next_high[:,0]>=5)&(next_high[:,0]<=6)&(next_high[:,1]<=.5)) | ((next_high[:,0]>6)&(next_high[:,1]<=1e-6))
    next_min=torch.where(all_safe_terminal,torch.full_like(next_min,32.),next_min)
    live=(states[:,0]>6)&(states[:,1]>1e-6)
    active=live&(h>=0)
    drift=(1-decay)*h-next_min
    q3=torch.where(active,drift,torch.zeros_like(drift))
    return torch.stack([-model(initial),model(unsafe)+model.delta,q3],1)


def scores(model,initial,unsafe,states,next_high,decay=.05):
    return raw_constraints(model,initial,unsafe,states,next_high,decay).amax(1).clamp(min=0)


def sample_panel(count,seed):
    rng=np.random.default_rng(seed)
    return tuple(torch.tensor(rng.uniform(lo,hi,size=(count,2)),dtype=torch.float32)
                 for lo,hi in [([15.,2.5],[16.,3.]),([5.,.5],[6.,3.]),([5.,0.],[16.,3.])])
