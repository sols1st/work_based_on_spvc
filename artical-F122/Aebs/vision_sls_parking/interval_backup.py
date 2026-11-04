"""Outward interval RK4 and finite-horizon braking feasibility.

All interval elementary arithmetic is rounded outwards; trigonometric ranges
use mpmath interval arithmetic. The mathematical model is the registered RK4
plant with zero process noise, optionally plus a declared numerical allowance.
This does not certify a particular platform's libm implementation.
"""
import numpy as np
from mpmath import iv
from .core import DT, STATE_LOW, STATE_HIGH, OBSTACLES
from .geometric_neural_sbc import zero_region

iv.dps = 35


def point(x):
    x = np.asarray(x, dtype=np.float64)
    return x.copy(), x.copy()


def add(a,b):
    return np.nextafter(a[0]+b[0],-np.inf),np.nextafter(a[1]+b[1],np.inf)


def mul(a,b):
    products=np.stack([a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]])
    return np.nextafter(products.min(0),-np.inf),np.nextafter(products.max(0),np.inf)


def trig(a,cosine=False):
    low=np.empty_like(a[0]);high=np.empty_like(a[1]);fn=iv.cos if cosine else iv.sin
    # Identical heading intervals across hypotheses need only one evaluation.
    cache={}
    for index in np.ndindex(low.shape):
        pair=(float(a[0][index]),float(a[1][index]))
        if pair not in cache:
            value=fn(iv.mpf(pair))
            cache[pair]=(np.nextafter(float(value.a),-np.inf),np.nextafter(float(value.b),np.inf))
        low[index],high[index]=cache[pair]
    return low,high


def step(box,action,allowance=1e-12):
    low,high=np.asarray(box[0]),np.asarray(box[1]);u=np.asarray(action,dtype=np.float64)
    if (low.shape!=high.shape or low.ndim!=2 or low.shape[1]!=4 or (low>high).any()
            or not np.isfinite(np.r_[low.ravel(),high.ravel(),u]).all() or u.shape!=(2,)
            or not np.isfinite(allowance) or allowance<0):
        raise ValueError('finite ordered state boxes and a finite action required')
    def ode(q):
        velocity=q[0][:,3],q[1][:,3];heading=q[0][:,2],q[1][:,2]
        xdot=mul(velocity,trig(heading,True));ydot=mul(velocity,trig(heading))
        return (np.column_stack([xdot[0],ydot[0],np.full(len(low),u[0]),np.full(len(low),u[1])]),
                np.column_stack([xdot[1],ydot[1],np.full(len(low),u[0]),np.full(len(low),u[1])]))
    x=(low,high);k1=ode(x);k2=ode(add(x,mul(point(DT*.5),k1)))
    k3=ode(add(x,mul(point(DT*.5),k2)));k4=ode(add(x,mul(point(DT),k3)))
    slope=add(add(k1,mul(point(2.),k2)),add(mul(point(2.),k3),k4))
    # Enclose DT/6 instead of treating its rounded binary result as exact.
    factor=(np.nextafter(DT/6,-np.inf),np.nextafter(DT/6,np.inf))
    result=add(x,mul(factor,slope))
    return add(result,(-allowance,allowance))


def safe(box):
    lo,hi=box
    if not np.isfinite(np.r_[lo.ravel(),hi.ravel()]).all() or (lo>hi).any():
        return False
    if (lo<STATE_LOW-.001).any() or (hi>STATE_HIGH+.001).any():
        return False
    for px,py,r in OBSTACLES:
        center=np.array([px,py])
        below=np.nextafter(center-hi[:,:2],-np.inf)
        above=np.nextafter(lo[:,:2]-center,-np.inf)
        distance=np.maximum(np.maximum(below,above),0)
        square=np.nextafter(distance*distance,-np.inf)
        lower=np.nextafter(square.sum(1),-np.inf)
        threshold=np.nextafter(np.nextafter(r*r,np.inf)+.0001,np.inf)
        if (lower<=threshold).any():
            return False
    return zero_region(box)


def braking_action(box):
    # All hypotheses have the same known acceleration history. A residual
    # initial velocity interval is retained, never silently set to zero.
    center=(float(box[0][:,3].min())+float(box[1][:,3].max()))/2
    return np.array([0.,np.clip(-center/DT,-4.,4.)])


def backup(box,time,allowance=1e-12):
    if time not in range(31):
        raise ValueError('finite-horizon time required')
    if not safe(box):
        return False
    current=box
    for _ in range(time,30):
        current=step(current,braking_action(current),allowance)
        if not safe(current):
            return False
    return True


def sequence_safe(box,actions,allowance=1e-12):
    return sequence_box(box,actions,allowance) is not None


def sequence_box(box,actions,allowance=1e-12):
    if not safe(box):return None
    for action in actions:
        box=step(box,action,allowance)
        if not safe(box):return None
    return box
