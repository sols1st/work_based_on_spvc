"""Constructive nonnegative ReLU SBC plus a bounded learned correction.

Its drift proof uses the governor's verified geometric zero-set invariant.
The bounded learned term is retained from the prior trained SBC; it is not
credited with proving the invariant. This is a changed closed-loop design.
"""
import copy
import numpy as np
import torch
from torch import nn
from mpmath import iv
from .core import STATE_LOW,STATE_HIGH,OBSTACLES

MARGIN=.005
ANGLES=np.arange(32)*2*np.pi/32
NORMALS=np.column_stack([np.cos(ANGLES),np.sin(ANGLES)])
iv.dps=35
_norm=max(np.nextafter(float(iv.sqrt(iv.mpf(float(a))**2+iv.mpf(float(b))**2).b),np.inf) for a,b in NORMALS)
RADII=[]
for _r in OBSTACLES[:,2]:
    _square=np.nextafter(np.nextafter(_r*_r,np.inf)+.0001,np.inf)
    _bound=iv.sqrt(iv.mpf(float(_square)))*iv.mpf(float(_norm))
    RADII.append(np.nextafter(float(_bound.b),np.inf))
RADII=np.asarray(RADII)


def zero_region(box):
    """Outward sufficient enclosure of G(x)=0 over a whole state box."""
    lo,hi=box
    if (lo<np.nextafter(STATE_LOW-.001+MARGIN,np.inf)).any():return False
    if (hi>np.nextafter(STATE_HIGH+.001-MARGIN,-np.inf)).any():return False
    for center,radius in zip(OBSTACLES[:,:2],RADII):
        a=np.nextafter(lo[:,:2]-center,-np.inf);b=np.nextafter(hi[:,:2]-center,np.inf)
        products=np.minimum(a[:,None,:]*NORMALS[None],b[:,None,:]*NORMALS[None])
        products=np.nextafter(products,-np.inf)
        lower=np.nextafter(products.sum(2),-np.inf).max(1)
        if (lower<np.nextafter(radius+MARGIN,np.inf)).any():return False
    return True


def relu_max(values):
    """max via ReLU(a-b)+b: a fixed-weight ReLU computation graph."""
    result=values[...,0]
    for k in range(1,values.shape[-1]):
        candidate=values[...,k];result=torch.relu(result-candidate)+candidate
    return result


class GeometricNeuralSBC(nn.Module):
    def __init__(self,trained_barrier):
        super().__init__();self.learned=copy.deepcopy(trained_barrier)
        for name,value in [('normals',NORMALS),('radii',RADII),('centers',OBSTACLES[:,:2]),
                           ('low',STATE_LOW-.001),('high',STATE_HIGH+.001)]:
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))

    def geometric(self,state):
        x=state[...,:4].double()
        bound=torch.cat([1+(self.low-x)/MARGIN,1+(x-self.high)/MARGIN],dim=-1)
        support=(x[...,None,:2]-self.centers)@self.normals.T
        obstacle=1+(self.radii-relu_max(support))/MARGIN
        return 20*torch.relu(relu_max(torch.cat([bound,obstacle],dim=-1)))[...,None]

    def forward(self,state,reference,time):
        if state.ndim!=2 or state.shape[1]!=8 or (time<0).any() or (time>30).any():
            raise ValueError('physical + observer state and valid horizon time required')
        dtype=next(self.learned.parameters()).dtype
        features=self.learned.features(state.to(dtype=dtype),reference,time)
        raw=self.learned.net[:-1](features)
        correction=.005*torch.sigmoid(raw).double()
        return self.geometric(state)+.01*(30-time[:,None]).double()+correction
