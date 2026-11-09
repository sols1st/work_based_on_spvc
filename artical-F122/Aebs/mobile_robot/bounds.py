"""Three-state/two-action interval adapters; normalized state coordinates.

Float64 with numerical padding, not directed-rounding formal arithmetic.
"""
import itertools
import numpy as np
import torch
from Aebs.vision_sls.neural_bounds import tanh_affine_bounds
from . import core


def grid_boxes(low,high,counts,dtype=torch.float64):
    if isinstance(counts,int):counts=[counts]*3
    cuts=[torch.linspace(float(a),float(b),n+1,dtype=dtype) for a,b,n in zip(low,high,counts)]
    lo=torch.stack(torch.meshgrid(*[c[:-1] for c in cuts],indexing='ij'),-1).reshape(-1,3)
    hi=torch.stack(torch.meshgrid(*[c[1:] for c in cuts],indexing='ij'),-1).reshape(-1,3)
    return lo,hi


def error_boxes(radius,splits=2,dtype=torch.float64):
    radii=np.broadcast_to(np.asarray(radius,dtype=float),(3,))
    if not np.isfinite(radii).all() or (radii<0).any():raise ValueError('finite nonnegative radii required')
    lo,hi=grid_boxes(-radii,radii,splits,dtype)
    return list(zip(lo,hi))


def multiply(al,ah,bl,bh):
    vals=torch.stack([al*bl,al*bh,ah*bl,ah*bh])
    return vals.amin(0),vals.amax(0)


def trig_bounds(lo,hi,cos=False):
    # General periodic bounds, including intervals crossing critical points.
    phase=np.pi/2 if cos else 0.
    lo,hi=lo+phase,hi+phase
    lower=torch.minimum(lo.sin(),hi.sin());upper=torch.maximum(lo.sin(),hi.sin())
    has_max=torch.ceil((lo-np.pi/2)/(2*np.pi))<=torch.floor((hi-np.pi/2)/(2*np.pi))
    has_min=torch.ceil((lo+np.pi/2)/(2*np.pi))<=torch.floor((hi+np.pi/2)/(2*np.pi))
    return torch.where(has_min,-torch.ones_like(lower),lower),torch.where(has_max,torch.ones_like(upper),upper)


def transitions(actor,lo,hi,el,eh):
    al,ah=tanh_affine_bounds(list(actor.policy_net)+[actor.action_net],lo+el,hi+eh)
    al,ah=al.clamp(-1,1),ah.clamp(-1,1)
    vl,vh=(al[:,0]+1)*core.V_MAX/2,(ah[:,0]+1)*core.V_MAX/2
    sl,sh=trig_bounds(lo[:,2]*core.SCALE[2],hi[:,2]*core.SCALE[2])
    cl,ch=trig_bounds(lo[:,2]*core.SCALE[2],hi[:,2]*core.SCALE[2],cos=True)
    dxl,dxh=multiply(vl,vh,cl,ch);dyl,dyh=multiply(vl,vh,sl,sh)
    dl=torch.stack([dxl/core.SCALE[0],dyl/core.SCALE[1],al[:,1]*core.OMEGA_MAX/core.SCALE[2]],1)*core.DT
    dh=torch.stack([dxh/core.SCALE[0],dyh/core.SCALE[1],ah[:,1]*core.OMEGA_MAX/core.SCALE[2]],1)*core.DT
    return lo+dl-1e-12,hi+dh+1e-12


def obstacle_boxes(grid=12,dtype=torch.float64):
    low=core.normalize([*(core.OBSTACLE-core.EXCLUSION_RADIUS),core.LOW[2]-.1])
    high=core.normalize([*(core.OBSTACLE+core.EXCLUSION_RADIUS),core.HIGH[2]+.1])
    lo,hi=grid_boxes(low,high,[grid,grid,8],dtype)
    # Every box intersecting the unsafe closed disk, not just its center.
    pl=lo[:,:2]*lo.new_tensor(core.SCALE[:2])+lo.new_tensor(core.CENTER[:2]-core.OBSTACLE)
    ph=hi[:,:2]*hi.new_tensor(core.SCALE[:2])+hi.new_tensor(core.CENTER[:2]-core.OBSTACLE)
    nearest=torch.maximum(torch.maximum(pl,-ph),torch.zeros_like(pl))
    mask=nearest.square().sum(1)<=core.EXCLUSION_RADIUS**2+1e-12
    return lo[mask],hi[mask]


def unsafe_boxes(dtype=torch.float64):
    # Any one-step successor of a safe state lies in this enlarged box.
    reach=np.array([core.DT*core.V_MAX,core.DT*core.V_MAX,core.DT*core.OMEGA_MAX])
    outer_lo,outer_hi=core.normalize(core.LOW-reach),core.normalize(core.HIGH+reach)
    safe_lo,safe_hi=core.normalize(core.LOW),core.normalize(core.HIGH)
    pairs=[]
    for axis in range(3):
        lo,hi=outer_lo.copy(),outer_hi.copy();hi[axis]=safe_lo[axis]
        pairs.append(grid_boxes(lo,hi,[16 if d!=axis else 1 for d in range(3)],dtype))
        lo,hi=outer_lo.copy(),outer_hi.copy();lo[axis]=safe_hi[axis]
        pairs.append(grid_boxes(lo,hi,[16 if d!=axis else 1 for d in range(3)],dtype))
    pairs.append(obstacle_boxes(dtype=dtype))
    return torch.cat([p[0] for p in pairs]),torch.cat([p[1] for p in pairs])


def wholly_unsafe(lo,hi):
    sl=lo.new_tensor(core.normalize(core.LOW));sh=lo.new_tensor(core.normalize(core.HIGH))
    outside=((hi<sl)|(lo>sh)).any(1)
    pl=lo[:,:2]*lo.new_tensor(core.SCALE[:2])+lo.new_tensor(core.CENTER[:2]-core.OBSTACLE)
    ph=hi[:,:2]*hi.new_tensor(core.SCALE[:2])+hi.new_tensor(core.CENTER[:2]-core.OBSTACLE)
    disk=torch.maximum(pl.abs(),ph.abs()).square().sum(1)<=core.EXCLUSION_RADIUS**2
    return outside|disk


def bisect(lo,hi):
    axis=(hi-lo).argmax(1);row=torch.arange(len(lo));mid=(lo+hi)/2
    upper=hi.clone();upper[row,axis]=mid[row,axis]
    lower=lo.clone();lower[row,axis]=mid[row,axis]
    return torch.cat([lo,lower]),torch.cat([upper,hi])
