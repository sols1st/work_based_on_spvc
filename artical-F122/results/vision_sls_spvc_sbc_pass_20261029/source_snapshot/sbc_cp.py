"""Observation-tube checks for the ORIGINAL SPVC nonnegative SBC.

No signed NCBF, action projection, or replacement barrier is constructed here.
A conformal interval alone does not specify a conditional noise distribution.
The worst-case successor value below can upper-bound an expectation ONLY if
that conditional distribution has support entirely inside the observation tube.
"""
import torch
from torch.nn import functional as F
from Aebs.VT.utils import MLP
from .cp_ncbf import action_bounds, successors  # generic actor/plant bounds only


class NonnegativeSBC(MLP):
    """Original tanh MLP with an explicit nonnegative output activation.

    ReLU is continuous and Lipschitz and allows an exactly zero safe interior.
    It retains the nonnegative SBC conditions, not the signed NCBF convention.
    Record the activation with checkpoints; old checkpoints default to softplus.
    """
    def __init__(self,features=(2,16,8,1),nonnegative_activation='softplus'):
        super().__init__(list(features),activation='tanh',square_output=True)
        if nonnegative_activation not in ('softplus','relu'):raise ValueError('unknown nonnegative activation')
        self.nonnegative_activation=nonnegative_activation

    def forward(self,x):
        for layer in self.layers[:-1]:x=layer(x).tanh()
        x=self.layers[-1](x)
        return x.relu() if self.nonnegative_activation=='relu' else F.softplus(x)


def load_sbc_checkpoint(path,config=None):
    """Load activation-aware SBCs; legacy raw weights require their config.

    ReLU and softplus share parameter names but define different certificates.
    The published format records this distinction rather than guessing it.
    """
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('format')=='vision_sls_nonnegative_sbc_v2':
        metadata=saved['metadata'];state=saved['state_dict']
        if config is not None and config.get('nonnegative_activation',metadata['nonnegative_activation'])!=metadata['nonnegative_activation']:
            raise ValueError('SBC activation does not match checkpoint metadata')
        model=NonnegativeSBC(saved['features'],metadata['nonnegative_activation'])
    else:
        if config is None:raise ValueError('legacy raw SBC weights require explicit activation and scale config')
        metadata=config;state=saved
        model=NonnegativeSBC(config.get('features',[2,16,8,1]),config['nonnegative_activation'])
    model.load_state_dict(state)
    return model,metadata


def barrier_bounds(barrier, lower, upper):
    """IBP through the actual VT.utils.MLP, including its softplus output.

    Despite its historical name, square_output=True uses softplus, not squaring.
    Arithmetic is numerical IBP, without formal outward floating-point rounding.
    """
    if lower.shape != upper.shape or lower.ndim != 2 or not (lower <= upper).all():
        raise ValueError('ordered matching [batch,2] interval endpoints required')
    for index, layer in enumerate(barrier.layers):
        positive, negative = layer.weight.clamp(min=0), layer.weight.clamp(max=0)
        lower, upper = (F.linear(lower, positive, layer.bias) + F.linear(upper, negative),
                        F.linear(upper, positive, layer.bias) + F.linear(lower, negative))
        if index < len(barrier.layers)-1:
            if barrier.activation == 'tanh': lower, upper = lower.tanh(), upper.tanh()
            elif barrier.activation == 'relu': lower, upper = lower.relu(), upper.relu()
            else: raise ValueError('unsupported original SBC activation')
    if barrier.square_output:
        if getattr(barrier,'nonnegative_activation','softplus')=='relu':lower,upper=lower.relu(),upper.relu()
        else:lower,upper=F.softplus(lower),F.softplus(upper)
    return lower.flatten(), upper.flatten()


@torch.no_grad()
def sbc_observation_tube_audit(barrier, policy, states, radii, barrier_distance_scale,
                              reach_probability=.95, epsilon=.1):
    """Check all supplied states required by the original threshold rule.

    Positive IBP upper residuals mean unresolved sufficient conditions, not
    necessarily real violations. A separate nominal transition gives witnesses.
    No 0.95*unsafe_min band filter and no historical 0.1% violation tolerance.
    """
    if not 0 < reach_probability < 1 or epsilon <= 0 or barrier_distance_scale <= 0:
        raise ValueError('valid probability, strict drift and scale required')
    scale = torch.tensor([barrier_distance_scale, 1.], dtype=states.dtype)
    actions = action_bounds(policy, states, radii)
    lower, upper = successors(states, actions)
    _, next_upper = barrier_bounds(barrier, lower/scale, upper/scale)
    current = barrier(states/scale).flatten()
    threshold = 1/(1-reach_probability)
    unsafe = (states[:,0] <= 6)&(states[:,1] >= .5)
    active = (~unsafe)&(current <= threshold)
    robust_residual = next_upper + epsilon - current
    nominal_action = policy.forward_observation(states[:,:1]/policy.distance_scale_m, states[:,1:]).flatten()
    nominal_next, _ = successors(states, torch.stack([nominal_action, nominal_action], 1))
    nominal_residual = barrier(nominal_next/scale).flatten() + epsilon - current
    exits = (lower[:,0]<5)|(upper[:,0]>16)|(lower[:,1]<0)|(upper[:,1]>3)
    return dict(active=active, current=current, robust_residual=robust_residual,
                nominal_residual=nominal_residual, successor_domain_exit=exits,
                action_intervals=actions, successor_low=lower, successor_high=upper)


def _product_bounds(a,b,c,d):
    values=torch.stack([a*c,a*d,b*c,b*d])
    return values.amin(0),values.amax(0)


def velocity_derivative_bounds(barrier,lower,upper,coordinate=1):
    """Interval bound for dB/dv; useful to avoid dependency blowup in IBP."""
    dl=torch.zeros_like(lower);dl[:,coordinate]=1.;du=dl.clone()
    for index,layer in enumerate(barrier.layers):
        pos,neg=layer.weight.clamp(min=0),layer.weight.clamp(max=0)
        lower,upper=(F.linear(lower,pos,layer.bias)+F.linear(upper,neg),
                     F.linear(upper,pos,layer.bias)+F.linear(lower,neg))
        dl,du=(F.linear(dl,pos)+F.linear(du,neg),F.linear(du,pos)+F.linear(dl,neg))
        if index<len(barrier.layers)-1:
            if barrier.activation!='tanh': raise ValueError('derivative bounds currently support original tanh SBC')
            # tanh derivative is nonnegative, maximal at zero.
            far=torch.maximum(lower.abs(),upper.abs())
            near=torch.where((lower<=0)&(upper>=0),torch.zeros_like(lower),torch.minimum(lower.abs(),upper.abs()))
            slope_low=1-far.tanh().square();slope_high=1-near.tanh().square()
            dl,du=_product_bounds(dl,du,slope_low,slope_high)
            lower,upper=lower.tanh(),upper.tanh()
    if barrier.square_output:
        if getattr(barrier,'nonnegative_activation','softplus')=='relu':
            dl,du=_product_bounds(dl,du,(lower>0).to(lower.dtype),(upper>=0).to(upper.dtype))
        else:dl,du=_product_bounds(dl,du,lower.sigmoid(),upper.sigmoid())
    return dl.flatten(),du.flatten()


def monotone_box_bounds(barrier,lower,upper):
    """Tighten state-box bounds only when an interval derivative proves its sign."""
    min_lo=lower.clone();min_hi=upper.clone();max_lo=lower.clone();max_hi=upper.clone()
    for coordinate in range(lower.shape[1]):
        dl,du=velocity_derivative_bounds(barrier,lower,upper,coordinate)
        increasing=dl>=0;decreasing=du<=0
        min_hi[:,coordinate]=torch.where(increasing,lower[:,coordinate],min_hi[:,coordinate])
        max_lo[:,coordinate]=torch.where(increasing,upper[:,coordinate],max_lo[:,coordinate])
        min_lo[:,coordinate]=torch.where(decreasing,upper[:,coordinate],min_lo[:,coordinate])
        max_hi[:,coordinate]=torch.where(decreasing,lower[:,coordinate],max_hi[:,coordinate])
        # If both derivatives are zero the function is constant along this
        # coordinate; select one endpoint consistently (not an inverted box).
        both=increasing&decreasing
        min_lo[:,coordinate]=torch.where(both,lower[:,coordinate],min_lo[:,coordinate])
        min_hi[:,coordinate]=torch.where(both,lower[:,coordinate],min_hi[:,coordinate])
        max_lo[:,coordinate]=torch.where(both,upper[:,coordinate],max_lo[:,coordinate])
        max_hi[:,coordinate]=torch.where(both,upper[:,coordinate],max_hi[:,coordinate])
    lb,_=barrier_bounds(barrier,min_lo,min_hi);_,ub=barrier_bounds(barrier,max_lo,max_hi)
    return lb,ub


def successor_upper(barrier,lower,upper,subdivisions=16):
    """Tighter bound for a fixed next distance and uncertain next velocity.

    Proven monotonicity selects an endpoint. Other intervals are subdivided;
    each remaining cell uses IBP. Never replace a maximum by sampled values.
    """
    if subdivisions<1 or not torch.equal(lower[:,0],upper[:,0]):
        raise ValueError('fixed distance and positive subdivision count required')
    with torch.no_grad(): dl,du=velocity_derivative_bounds(barrier,lower,upper)
    increasing=dl>=0;decreasing=du<=0;ambiguous=~(increasing|decreasing)
    # ReLU backward retains its output; do not overwrite that storage when
    # replacing ambiguous intervals with their refined bounds.
    result=barrier(torch.where(increasing[:,None],upper,lower)).flatten().clone()
    if ambiguous.any():
        lo=lower[ambiguous];hi=upper[ambiguous]
        edges=torch.linspace(0,1,subdivisions+1,device=lo.device)
        endpoints=lo[:,None,:]+(hi-lo)[:,None,:]*edges[None,:,None]
        ll=endpoints[:,:-1].reshape(-1,2);uu=endpoints[:,1:].reshape(-1,2)
        with torch.no_grad(): sl,su=velocity_derivative_bounds(barrier,ll,uu)
        _,bound=barrier_bounds(barrier,ll,uu)
        bound=torch.where(sl>=0,barrier(uu).flatten(),bound)
        bound=torch.where(su<=0,barrier(ll).flatten(),bound)
        result[ambiguous]=bound.reshape(len(lo),subdivisions).amax(1)
    return result
