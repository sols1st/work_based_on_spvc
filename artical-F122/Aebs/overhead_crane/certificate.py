"""Finite-horizon robust barrier from support functions of the closed-loop model."""
from fractions import Fraction
from itertools import product
import numpy as np
from . import plant

STATE_ROWS = {
    'trolley_position':(1,0,0,0),'trolley_speed':(0,1,0,0),
    'sway':(0,0,1,0),'sway_rate':(0,0,0,1),
    'payload_position':(1,0,1,0),'payload_diagonal':(1,0,1,0)}


def planes(gains,q,horizon,exact):
    # Each yield is one term in the maximum defining B_t. A repeated signed
    # row covers the diagonal load clearance or the shared motor budget.
    if not np.isfinite(q) or q<0 or not np.isfinite(gains).all():
        raise ValueError('finite gains and finite nonnegative CP radius required')
    def number(x):
        return Fraction(str(x)) if exact else float(x)
    radius=Fraction.from_float(float(q)) if exact else float(q)
    dt,g,drag=map(number,[plant.DT,plant.GRAVITY_OVER_LENGTH,plant.DAMPING])
    a,b,wind=map(np.array,plant.coefficients(dt,g,drag))
    k=np.array([number(v) for v in gains])
    closed=a-np.outer(b,k)
    # Observation errors affect only trolley position and relative displacement.
    action_error=(abs(k[0])+number(.5)*abs(k[2]))*radius
    drive_noise=action_error+number(plant.CART_NOISE)
    gust_noise=number(plant.LOAD_NOISE)
    rows={**STATE_ROWS,'axis_accel':k,'total_accel':k}
    limits={**plant.LIMITS,'axis_accel':plant.AXIS_ACCEL,'total_accel':plant.TOTAL_ACCEL}
    for name,base in rows.items():
        motor=name in ('axis_accel','total_accel')
        row=np.array([number(v) for v in base])
        offset=action_error if motor else 0
        for delay in range(horizon+(0 if motor else 1)):
            yield name,delay,row,offset,number(limits[name])
            offset+=abs(row@b)*drive_noise+abs(row@wind)*gust_noise
            row=row@closed


def verify(gains,q,exact=True):
    box=np.array([Fraction(str(v)) if exact else v for v in plant.INITIAL])
    peaks={}
    for name,delay,row,offset,limit in planes(gains,q,plant.HORIZON,exact):
        axes=2 if name in ('total_accel','payload_diagonal') else 1
        value=axes*(abs(row)@box+offset)/limit
        if name not in peaks or value>peaks[name][0]:
            peaks[name]=(value,delay)
    largest=max(v for v,_ in peaks.values())
    return dict(passed=bool(largest<1),initial_upper=float(largest),unsafe_lower=1,
                mode='exact rational' if exact else 'development float',
                initial_upper_exact=str(largest) if exact else None,
                worst_ratio={k:dict(ratio=float(v),step=j) for k,(v,j) in peaks.items()})


def barrier(state,gains,q,remaining):
    """Evaluate the explicit time-indexed function on a batch of two-axis states."""
    if not 0<=remaining<=plant.HORIZON:
        raise ValueError('invalid remaining horizon')
    value=np.zeros(state.shape[:-2])
    for name,_,row,offset,limit in planes(gains,q,remaining,False):
        projected=state@row
        if name=='total_accel':
            bound=np.abs(projected).sum(axis=-1)+2*offset
        elif name=='payload_diagonal':
            bound=np.abs(projected.sum(axis=-1))+2*offset
        else:
            bound=np.abs(projected).max(axis=-1)+offset
        value=np.maximum(value,bound/limit)
    return value


def lqr(weights,action_cost):
    a,b,_=plant.matrices()
    cost=np.diag(weights)
    p=cost.copy()
    for _ in range(500):
        k=(b@p@a)/(action_cost+b@p@b)
        nxt=cost+a.T@p@a-np.outer(a.T@p@b,k)
        if np.abs(nxt-p).max()<1e-12:
            break
        p=nxt
    return np.round(k,4)


def synthesize():
    init,noise,_=plant.scenarios(128,81119)
    design_q=.12
    rng=np.random.default_rng(81120)
    bias=rng.choice([-design_q,design_q],size=(len(init),4))
    errors=np.concatenate([np.zeros_like(bias),bias,-bias])
    starts=np.tile(init,(3,1,1))
    disturbances=np.tile(noise,(3,1,1,1))
    candidates=[]
    for swing,effort in product([2.,6.,14.,28.],[.5,1.,2.,4.]):
        k=lqr([5.,.8,swing,.6],effort)
        audit=verify(k,design_q,False)
        state=starts.copy()
        loss=np.zeros(len(state))
        for i in range(plant.HORIZON):
            action=plant.control(plant.observation(state)+errors,state,k)
            state=plant.step(state,action,disturbances[:,i])
            loss+=((state[...,0]+state[...,2])**2+3*state[...,2]**2+.05*action**2).sum(-1)
        cost=float(loss.reshape(3,-1).max(0).mean())
        candidates.append(dict(weights=[5.,.8,swing,.6],effort=effort,gains=k.tolist(),cost=cost,
                               development_ratio=audit['initial_upper'],feasible=audit['passed']))
    feasible=[row for row in candidates if row['feasible']]
    if not feasible:
        raise ValueError(f'no feasible design; best ratio {min(r["development_ratio"] for r in candidates)}')
    winner=min(feasible,key=lambda row:row['cost'])
    return dict(design_radius=design_q,candidates=candidates,selected=winner,
                feasible=len(feasible),all=len(candidates))
