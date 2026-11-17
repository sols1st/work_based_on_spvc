"""Time-varying support-function certificate for an eight-state crane with CP observation boxes."""
import json
import math
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np

from ..conformal.trajectory_boxes import width_bounds
from . import certificate, plant

# Plant coefficients are exact decimal rationals. Kernel bounds and width propagation
# use upward-rounded fixed point integers; this keeps products well within int64.
UNIT = 10**6
FEATURE_CELL = UNIT // 40
ROW_NAMES = ('trolley_position','trolley_speed','sway','sway_rate','payload_position','motor')
GAIN_FILE = Path(__file__).resolve().parents[2] / 'results/overhead_crane_8d_20261117_v1/evaluation/synthesis.json'


def features(states):
    x = np.asarray(states)
    return np.stack([np.max(np.abs(x[...,0]), axis=-1),
                     2*np.max(np.abs(x[...,2]), axis=-1)], axis=-1)


def exact(x):
    return Fraction.from_float(float(x))


def exp_upper(x):
    # Decimal.exp is correctly rounded to nearest. Move one representable number up.
    with localcontext() as ctx:
        ctx.prec = 42
        ctx.rounding = ROUND_CEILING
        exponent = Decimal(x.numerator) / Decimal(x.denominator)
        return Fraction(exponent.exp().next_plus())


def mathematical_widths(shape, lower, upper):
    """Upper bounds on the declared real interpolant and on the implemented box."""
    flo = np.nextafter([float(v) for v in lower], -np.inf)
    fhi = np.nextafter([float(v) for v in upper], np.inf)
    widths = np.array([[exact(v) for v in sign] for sign in width_bounds(shape,flo,fhi)],dtype=object)
    name = shape['name']
    if name == 'state_interp':
        knots = list(map(exact, shape['centers']))
        values = np.vectorize(exact, otypes=[object])(np.asarray(shape['halfwidths'])[:,0])
        def at(x):
            if x <= knots[0]:
                return values[0]
            if x >= knots[-1]:
                return values[-1]
            i = next(i for i in range(len(knots)-1) if knots[i] <= x <= knots[i+1])
            w = (x-knots[i]) / (knots[i+1]-knots[i])
            return values[i]*(1-w) + values[i+1]*w
        for x in [lower[0],upper[0],*(x for x in knots if lower[0]<=x<=upper[0])]:
            widths = np.maximum(widths, at(x))
    if name == 'state_loglinear':
        reg = shape['regression']
        mins, maxs = [list(map(exact,reg[k])) for k in ('low','high')]
        u = []
        for j in range(2):
            span = max(maxs[j]-mins[j],exact(1e-10))
            u.append([(min(max(x,mins[j]),maxs[j])-mins[j])/span-Fraction(1,2)
                      for x in (lower[j],upper[j])])
        for d,base in enumerate(reg['base']):
            c = [exact(row[d]) for row in reg['coefficients']]
            top = c[0] + sum(max(v*c[j+1] for v in u[j]) for j in range(2))
            widths[:,d] = np.maximum(widths[:,d],exact(base)*exp_upper(min(max(top,-3),2)))
    return widths


@lru_cache(None)
def kernels():
    k = np.array([Fraction(str(v)) for v in json.loads(GAIN_FILE.read_text())['selected']['gains']])
    dt,gravity,drag = map(lambda x:Fraction(str(x)),(plant.DT,plant.GRAVITY_OVER_LENGTH,plant.DAMPING))
    a,b,w = map(np.array,plant.coefficients(dt,gravity,drag))
    closed = a - np.outer(b,k)
    rows = np.array([certificate.STATE_ROWS[n] for n in ROW_NAMES[:-1]] + [list(-k)],dtype=object)
    init = np.array([Fraction(str(v)) for v in plant.INITIAL])
    noise = np.array([Fraction(0)]*len(rows))
    pos,neg,offsets,initial = [],[],[],[]
    # Split initial p_x and p_y each at +/- INITIAL[0]/3, covering the entire original box.
    center_factors = np.array(list(product([-Fraction(2,3),Fraction(0),Fraction(2,3)],repeat=2)))
    radius = init.copy()
    radius[0] /= 3
    ceil_units = np.vectorize(lambda x: math.ceil(x*UNIT),otypes=[np.int64])
    for t in range(plant.HORIZON+1):
        slopes = abs(rows) @ radius
        centers = center_factors[:,:,None]*init[0]*rows[:,0]
        initial.append(np.stack([ceil_units(slopes+centers),ceil_units(slopes-centers)]))
        offsets.append(ceil_units(noise))
        impulse = rows @ b
        pos.append(ceil_units(np.maximum(impulse,0)))
        neg.append(ceil_units(np.maximum(-impulse,0)))
        noise = noise + abs(impulse)*Fraction(str(plant.CART_NOISE)) + abs(rows@w)*Fraction(str(plant.LOAD_NOISE))
        rows = rows @ closed
    p, n = np.array(pos),np.array(neg)
    return k,np.array(initial),np.array(offsets),p,n


def command_enclosure(shape, q, k):
    @lru_cache(None)
    def cached(pl,sl,ph,sh):
        # Expand the reachable feature rectangle out to a 0.025-wide lattice.
        lower = [Fraction(int(pl),40),Fraction(int(sl),40)]
        upper = [Fraction(int(ph),40),Fraction(int(sh),40)]
        widths = mathematical_widths(shape,lower,upper)
        # Returned widths are physical input bounds to a mathematical plant. One
        # extra 1e-6 acceleration unit also covers the last numerical width evaluation.
        positive, negative = [], []
        for axis in range(2):
            hi = lo = Fraction(0)
            for idx,coef in ((axis,-k[0]),(2+axis,-k[2]/2)):
                hi += abs(coef)*widths[int(coef>0),idx]
                lo += abs(coef)*widths[int(coef<0),idx]
            positive.append(math.ceil(UNIT*exact(q)*hi)+1)
            negative.append(math.ceil(UNIT*exact(q)*lo)+1)
        return np.array([positive,negative],dtype=np.int64)
    return cached


def ceil_div(a,b):
    return -((-a)//b)


def verify_scale(shape, centiscale, calc):
    k,initial,offsets,positive,negative = kernels()
    nbox = initial.shape[2]
    history = np.zeros((plant.HORIZON,2,nbox,2),dtype=np.int64)
    peak = 0.
    active = None
    passed = True
    limit = np.array([round(UNIT*plant.LIMITS[n]) for n in ROW_NAMES[:-1]])
    full_pos=[]; full_neg=[]
    if positive.max()>=20*UNIT or negative.max()>=20*UNIT:
        raise OverflowError('Impulse exceeded checked int64 range')
    for t in range(plant.HORIZON+1):
        # shape [upper / minus lower, initial sub-box, axis, output row].
        support = ceil_div(initial[t]*centiscale,100) + offsets[t]
        if t:
            # max abs(history) <= 20*UNIT and max abs(kernel) <= 20*UNIT.
            # Each sum has at most 80 terms, bounded by 3.2e16 < int64 maximum.
            hp,hn = np.moveaxis(history[:t],1,0)
            vals=[]
            for h1,h2 in ((hp,hn),(hn,hp)):
                v = np.einsum('tba,tr->bar',h1,positive[t-1::-1]) + np.einsum('tba,tr->bar',h2,negative[t-1::-1])
                vals.append(ceil_div(v,UNIT))
            support += np.stack(vals)
        smax = support.max(axis=0)
        state_max = smax[:,:,:-1].max(axis=(0,1))
        # Diagonal clearance |load_x+load_y|; keep both signs separately.
        diag = support[:,:,:,-2].sum(axis=2).max()
        peaks = list(zip(ROW_NAMES[:-1],state_max,limit))
        peaks.append(('payload_diagonal',diag,round(UNIT*plant.LIMITS['payload_diagonal'])))
        if t < plant.HORIZON:
            # Enclose max|p| and 2*max|sway| on each complete reachable coordinate box.
            j=[0,2]
            fhi = smax[:,:,j].max(axis=1)*[1,2]
            flo = np.maximum(0,-support[:,:,:,j].min(axis=0)).max(axis=1)*[1,2]
            bins_lo = flo//FEATURE_CELL
            bins_hi = ceil_div(fhi,FEATURE_CELL)
            if shape['name'].startswith('fixed_'):
                command = np.broadcast_to(calc(0,0,0,0)[:,None,:],(2,nbox,2))
            else:
                command = np.stack([calc(*a,*b) for a,b in zip(bins_lo,bins_hi)],axis=1)
            if np.max(command)>=20*UNIT:
                raise OverflowError('Command bound exceeded checked integer range')
            history[t] = command
            motor = (support[:,:,:,-1]+command).max(axis=0)
            peaks.extend([('axis_accel',motor.max(),round(UNIT*plant.AXIS_ACCEL)),
                          ('total_accel',motor.sum(axis=1).max(),round(UNIT*plant.TOTAL_ACCEL))])
            full_pos.append(command[0].max(axis=0))
            full_neg.append(command[1].max(axis=0))
        for name,value,bound in peaks:
            passed = passed and bool(value<bound)
            if value/bound > peak:
                peak,active=float(value/bound),[name,t]
    return dict(passed=passed, scale=centiscale/100, worst_ratio=peak, limiting=active,
                max_positive_action_error=(np.max(full_pos,axis=0)/UNIT).tolist(),
                max_negative_action_error=(np.max(full_neg,axis=0)/UNIT).tolist())


def audit(shape,q):
    """Return an attained (not critical) scale on a 0.01 grid with a fixed tube."""
    if not np.isfinite(q) or q<0:
        raise ValueError('Invalid conformal scale')
    calc=command_enclosure(shape,q,kernels()[0])
    nominal=verify_scale(shape,100,calc)
    # The search only attributes claims to the explicitly verified passing scale.
    lo,hi=(100,201) if nominal['passed'] else (0,100)
    if not nominal['passed'] and not verify_scale(shape,0,calc)['passed']:
        return dict(passed=False,certification_metric=0.,unit='initial_box_scale',nominal=nominal,
                    certified_scale_attained=False)
    while hi-lo>1:
        mid=(lo+hi)//2
        if verify_scale(shape,mid,calc)['passed']:
            lo=mid
        else:
            hi=mid
    final=verify_scale(shape,lo,calc)
    return dict(passed=nominal['passed'],certification_metric=lo/100,unit='initial_box_scale',
                nominal=nominal,accepted_scale=final,certified_scale_attained=final['passed'],
                search_grid=.01,search_max=2.,initial_partition='3x3 trolley initial coordinates',
                certificate='directional support of closed-loop reachable sets at every step; no saturation on passed domain')
