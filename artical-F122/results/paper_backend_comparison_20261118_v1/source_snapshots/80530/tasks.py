"""Frozen-task adapters for controlled backend comparisons.

These are NOT native SPVC/SPVT retraining. The actual frozen controller and
registered semantic CP box are shared across backends. A step uses the nominal
state-side observation plus a semantic residual, without executing DINO again.
All numerical interval routines are float64, WITHOUT directed rounding. The
structured hover/crane baseline separately reruns rational inequalities.

Public API: load_task(name), attributes documented on Task, scalar or batched
point steps/B evaluations, single-box interval steps and batched B bounds.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
NAMES = ('aebs', 'lane', 'braking', 'hover', 'crane')


def _json(path):
    return json.loads(Path(path).read_text())


def _binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _lin(matrix, lo, hi, bias=0.):
    pos, neg = np.maximum(matrix, 0), np.minimum(matrix, 0)
    return lo @ pos.T + hi @ neg.T + bias, hi @ pos.T + lo @ neg.T + bias


def _mul(a, b):
    v = np.stack([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return v.min(0), v.max(0)


def _abs(lo, hi):
    return np.where((lo <= 0) & (hi >= 0), 0., np.minimum(abs(lo), abs(hi))), np.maximum(abs(lo), abs(hi))


def _trig(lo, hi, cosine=False):
    # Explicit extrema, not monotonicity assumptions outside the task domain.
    if hi-lo >= 2*np.pi:
        return -1., 1.
    shift = 0. if cosine else np.pi/2
    f = np.cos if cosine else np.sin
    values = [f(lo), f(hi)]
    values += [f(shift+j*np.pi) for j in range(int(np.ceil((lo-shift)/np.pi)), int(np.floor((hi-shift)/np.pi))+1)]
    return min(values), max(values)


def _torch():
    import torch
    return torch


def _tensor(x):
    return _torch().as_tensor(np.asarray(x), dtype=_torch().float64)


def _np(x):
    return x.detach().cpu().numpy()


def _network_bounds(layers, lo, hi):
    # Reuses the project's documented numerical IBP, with a neural activation
    # wrapper that never samples a nonlinear extremum.
    from Aebs.vision_sls.cp_ncbf import sequential_bounds
    torch = _torch()
    with torch.no_grad():
        l, h = sequential_bounds(torch.nn.Sequential(*layers), _tensor(lo)[None], _tensor(hi)[None])
    return _np(l)[0], _np(h)[0]


@dataclass
class Task:
    name: str
    state_dim: int
    error_dim: int
    horizon: int
    dt: float
    initial_lo: np.ndarray
    initial_hi: np.ndarray
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    error_lo: np.ndarray
    error_hi: np.ndarray
    noise_lo: np.ndarray
    noise_hi: np.ndarray
    contexts: tuple = (None,)
    metadata: dict = field(default_factory=dict)

    def sample_initial(self, n, seed):
        rng = np.random.default_rng(seed)
        if hasattr(self, '_sample_initial'):
            return self._sample_initial(n, seed)
        return rng.uniform(self.initial_lo, self.initial_hi, (n, self.state_dim))

    def step(self, x, error=None, noise=None, context=None, t=0):
        a = np.asarray(x, dtype=float)
        one = a.ndim == 1
        a = np.atleast_2d(a)
        e = np.broadcast_to(np.zeros(self.error_dim) if error is None else error, (len(a), self.error_dim))
        w = np.broadcast_to(np.zeros(len(self.noise_lo)) if noise is None else noise, (len(a), len(self.noise_lo)))
        result = self._step(a, e, w, self.contexts[0] if context is None else context, t)
        return result[0] if one else result

    def nominal_step(self, x, context=None, t=0):
        return self.step(x, context=context, t=t)

    def interval_step(self, lo, hi, error_lo=None, error_hi=None, noise_lo=None, noise_hi=None, context=None, t=0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        if lo.shape != (self.state_dim,) or hi.shape != lo.shape or np.any(lo > hi):
            raise ValueError('interval_step requires one ordered state box')
        el = self.error_lo if error_lo is None else np.asarray(error_lo, float)
        eh = self.error_hi if error_hi is None else np.asarray(error_hi, float)
        wl = self.noise_lo if noise_lo is None else np.asarray(noise_lo, float)
        wh = self.noise_hi if noise_hi is None else np.asarray(noise_hi, float)
        if np.any(el > eh) or np.any(wl > wh):
            raise ValueError('inverted disturbance interval')
        return self._interval(lo, hi, el, eh, wl, wh, self.contexts[0] if context is None else context, t)

    def classify_safe(self, lo, hi=None, t=0):
        lo = np.asarray(lo, float)
        hi = lo if hi is None else np.asarray(hi, float)
        return self._safe(lo, hi, t)

    def barrier(self, x, t=0):
        a = np.asarray(x, dtype=float)
        value = self._barrier(np.atleast_2d(a), t)
        return float(value[0]) if a.ndim == 1 else value

    def barrier_bounds(self, lo, hi, t=0):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        single = lo.ndim == 1
        l, h = self._barrier_bounds(np.atleast_2d(lo), np.atleast_2d(hi), t)
        return (float(l[0]), float(h[0])) if single else (l, h)

    def is_terminal(self, x):
        if hasattr(self, '_terminal'):
            return self._terminal(np.asarray(x, float))
        return np.zeros(np.asarray(x).shape[:-1], dtype=bool)

    def current_baseline(self):
        return self._current_baseline()

    def current_verify(self, budget_s=300.):
        """Actually rerun checks; previous certificate reuse is labelled separately."""
        begin=time.monotonic()
        if hasattr(self,'_current_verify'):
            report=self._current_verify(float(budget_s))
        else:
            report=self._current_baseline()
        report.update(elapsed_seconds=time.monotonic()-begin,rerun=report.get('rerun',True),
                      comparison_scope='frozen current conditional robust backend',task=self.name)
        return report

    def affine_matrices(self, lo, hi, context=None, t=0):
        """Return x+=Ax+Be+Dw+c only on a proved no-saturation box."""
        return self._affine(np.asarray(lo),np.asarray(hi)) if hasattr(self,'_affine') else None

    def safe_linear_constraints(self, context=None, t=0):
        return getattr(self,'_safe_constraints',None)

    def affine_matrices_zonotope(self, center, generators, context=None, t=0):
        return self._affine_zonotope(np.asarray(center),np.asarray(generators)) if hasattr(self,'_affine_zonotope') else None


def _make(name, d, e, h, dt, il, ih, dl, dh, q, nl=(), nh=(), contexts=(None,)):
    q = np.broadcast_to(q, (e,)).copy()
    return Task(name,d,e,h,dt,*[np.asarray(v,float) for v in (il,ih,dl,dh,-q,q,nl,nh)],contexts,
                dict(comparison='controlled backend, frozen current visual interface/controller; not native retraining',
                     arithmetic='float64 interval arithmetic; no directed rounding',
                     barrier_nonnegative_by_construction=True, barrier_time_varying=False,
                     unsafe_barrier_threshold=20., frontend_execution='state proxy plus bounded semantic residual'))


def _rectangle_safe(lo, hi, lower, upper):
    if np.all(lo > lower) and np.all(hi < upper):
        return 'safe'
    if np.any(hi <= lower) or np.any(lo >= upper):
        return 'unsafe'
    return 'unknown'


def _neural_current(task, barrier, lower, upper, region_boxes, next_upper, budget_s):
    """Bound the entire original safe sublevel, never just sampled drift.

    This is a fresh bounded-budget numerical rerun. Timeout means unknown.
    Region and transition bounds reuse the original neural verifiers.
    """
    from Aebs.lane_keeping.learned_sbc import grid_boxes
    torch=_torch();start=time.monotonic();deadline=start+budget_s
    regions={};checked=0
    for name,(lo,hi,initial) in region_boxes.items():
        l,h=grid_boxes(lo,hi,32)
        accepted=0;pending=[(l,h,0)];worst=0. if initial else float('inf')
        while pending:
            a,b,depth=pending.pop()
            if time.monotonic()>deadline:
                return dict(passed=False,status='unknown',reason='time limit during region bounds',regions=regions,checked_cells=checked,rerun=True)
            with torch.no_grad():bl,bh=task._torch_barrier_bounds(a,b)
            good=bh<=1 if initial else bl>=20
            if good.any():worst=max(worst,float(bh[good].max())) if initial else min(worst,float(bl[good].min()))
            checked+=len(a);accepted+=int(good.sum())
            a,b=a[~good],b[~good]
            if len(a):
                if depth>=12 or len(a)>100000:
                    return dict(passed=False,status='unknown',reason='unresolved region',region=name,unresolved=len(a),checked_cells=checked,rerun=True)
                coordinate=(b-a).argmax(1);index=torch.arange(len(a));mid=(a+b)/2
                lh=b.clone();rl=a.clone();lh[index,coordinate]=mid[index,coordinate];rl[index,coordinate]=mid[index,coordinate]
                pending.extend([(torch.cat([a,rl]),torch.cat([lh,b]),depth+1)])
        regions[name]=dict(passed=True,bound=worst,accepted=accepted)
    contexts=[]
    for context in task.contexts:
        l,h=grid_boxes(lower,upper,32);queue=[(l,h,0)];accepted=excluded=0
        while queue:
            lo,hi,depth=queue.pop()
            lows=[];highs=[]
            for first in range(0,len(lo),256):
                if time.monotonic()>deadline:
                    return dict(passed=False,status='unknown',reason='global verification time limit',regions=regions,contexts=contexts,
                                current_context=context,checked_cells=checked,unresolved_at_least=len(lo)-first,rerun=True)
                a,b=lo[first:first+256],hi[first:first+256]
                with torch.no_grad():
                    bl,bh=task._torch_barrier_bounds(a,b)
                    inactive=bl>=20
                    nupper=next_upper(a,b,context)
                    good=(nupper<=bl)&~inactive
                accepted+=int(good.sum());excluded+=int(inactive.sum());checked+=len(a)
                bad=~(good|inactive)
                if bad.any():
                    # Concrete support point witnesses only disprove this B.
                    centers=_np((a[bad]+b[bad])/2)
                    now=task.barrier(centers)
                    for e in itertools.product(*zip(task.error_lo,task.error_hi)):
                        following=task.step(centers,np.asarray(e),context=context)
                        drift=task.barrier(following)-now
                        witness=(now<20)&(drift>1e-8)
                        if witness.any():
                            j=int(np.flatnonzero(witness)[0])
                            return dict(passed=False,status='witness',regions=regions,checked_cells=checked,context=context,
                                witness=dict(state=centers[j].tolist(),error=list(e),drift=float(drift[j])),rerun=True)
                    lows.append(a[bad]);highs.append(b[bad])
            if not lows:continue
            a,b=torch.cat(lows),torch.cat(highs)
            if depth>=16 or checked+2*len(a)>1000000:
                return dict(passed=False,status='unknown',reason='cell/depth budget',regions=regions,checked_cells=checked,unresolved=len(a),rerun=True)
            widths=(b-a)/torch.tensor(np.asarray(upper)-np.asarray(lower));coordinate=widths.argmax(1)
            index=torch.arange(len(a));mid=(a+b)/2;lh=b.clone();rl=a.clone()
            lh[index,coordinate]=mid[index,coordinate];rl[index,coordinate]=mid[index,coordinate]
            queue.append((torch.cat([a,rl]),torch.cat([lh,b]),depth+1))
        contexts.append(dict(context=context,passed=True,accepted=accepted,excluded=excluded))
    return dict(passed=True,status='pass',regions=regions,contexts=contexts,checked_cells=checked,rerun=True,
                arithmetic='float64 numerical intervals, without directed rounding')


def _hover(results):
    from Aebs.visual_hover import plant, certificate
    folder = results/'visual_hover_6d_20261117_v1/evaluation'
    design = _json(folder/'synthesis.json')['selected']
    q = _json(folder/'contract.json')['q']
    task = _make('hover',6,3,plant.HORIZON,plant.DT,[-plant.INIT_POS]*3+[-plant.INIT_VEL]*3,
                 [plant.INIT_POS]*3+[plant.INIT_VEL]*3,[-plant.POS_LIMIT]*3+[-plant.VEL_LIMIT]*3,
                 [plant.POS_LIMIT]*3+[plant.VEL_LIMIT]*3,q,[-plant.WIND]*3,[plant.WIND]*3)
    l,m,S = (design[k] for k in ('l','m','s_limit'))
    identity = np.eye(3)
    task.action_matrix = np.c_[-l*m*identity, -(l+m)*identity]
    task.closed_matrix = np.block([[identity,plant.DT*identity],[-plant.DT*l*m*identity,(1-plant.DT*(l+m))*identity]])
    task.error_matrix = np.r_[np.zeros((3,3)), -plant.DT*l*m*identity]
    task.noise_matrix = np.r_[np.zeros((3,3)), plant.DT*identity]
    def interval(lo,hi,el,eh,wl,wh,context,t):
        al,ah = _lin(task.action_matrix,lo,hi)
        al,ah = al-l*m*eh,ah-l*m*el
        if np.max(np.maximum(abs(al),abs(ah))) <= plant.ACCEL:
            nl,nh = _lin(task.closed_matrix,lo,hi)
            e1,e2 = _lin(task.error_matrix,el,eh)
            w1,w2 = _lin(task.noise_matrix,wl,wh)
            return nl+e1+w1,nh+e2+w2
        al,ah = np.clip(al,-plant.ACCEL,plant.ACCEL),np.clip(ah,-plant.ACCEL,plant.ACCEL)
        return np.r_[lo[:3]+plant.DT*lo[3:],lo[3:]+plant.DT*(al+wl)],np.r_[hi[:3]+plant.DT*hi[3:],hi[3:]+plant.DT*(ah+wh)]
    def bb(lo,hi,t):
        pl,ph=_abs(lo[:,:3],hi[:,:3]);sl,sh=_abs(lo[:,3:]+l*lo[:,:3],hi[:,3:]+l*hi[:,:3])
        lower=np.maximum(pl.max(1),sl.max(1)/S);upper=np.maximum(ph.max(1),sh.max(1)/S)
        return 20*np.maximum(0,(lower-1)/(plant.POS_LIMIT-1)),20*np.maximum(0,(upper-1)/(plant.POS_LIMIT-1))
    task._step=lambda x,e,w,c,t:plant.step(x,plant.control(x[:,:3]+e,x[:,3:],design),w)
    task._interval=interval
    task._safe=lambda lo,hi,t:_rectangle_safe(lo,hi,task.domain_lo,task.domain_hi)
    task._barrier=lambda x,t:certificate.barrier(x,design)
    task._barrier_bounds=bb
    task._current_baseline=lambda:certificate.verify(design,q)
    def affine(lo,hi):
        al,ah=_lin(task.action_matrix,lo,hi)
        if np.max(np.maximum(abs(al-l*m*task.error_hi),abs(ah-l*m*task.error_lo)))>plant.ACCEL:return None
        return task.closed_matrix,task.error_matrix,task.noise_matrix,np.zeros(6)
    task._affine=affine
    def affine_zonotope(center,generators):
        radius=np.abs(task.action_matrix@generators).sum(axis=1)+l*m*q
        if np.max(np.abs(task.action_matrix@center)+radius)>plant.ACCEL:return None
        return task.closed_matrix,task.error_matrix,task.noise_matrix,np.zeros(6)
    task._affine_zonotope=affine_zonotope
    task._safe_constraints=(np.r_[np.eye(6),-np.eye(6)],np.r_[task.domain_hi,-task.domain_lo])
    task.metadata.update(design=design,coordinates='three normalized positions then three normalized velocities',
                         contract=_binding(folder/'contract.json'),controller=_binding(folder/'synthesis.json'),
                         current_baseline_rerun='exact rational scalar inequalities')
    return task


def _crane(results):
    from Aebs.overhead_crane import plant, certificate
    folder=results/'overhead_crane_8d_20261117_v1/evaluation'
    k=np.asarray(_json(folder/'synthesis.json')['selected']['gains'])
    q=_json(folder/'contract.json')['q']
    half=np.tile(plant.INITIAL,2)
    lim=np.tile([1.45,1.8,.65,1.5],2)
    task=_make('crane',8,4,40,plant.DT,-half,half,-lim,lim,q,
                [-plant.CART_NOISE,-plant.LOAD_NOISE]*2,[plant.CART_NOISE,plant.LOAD_NOISE]*2)
    a,b,g=plant.matrices(); closed=a-np.outer(b,k)
    task.closed_matrix=np.kron(np.eye(2),closed)
    task.error_matrix=np.zeros((8,4));task.noise_matrix=np.zeros((8,4))
    for axis in range(2):
        task.error_matrix[4*axis:4*axis+4,axis]=-k[0]*b
        task.error_matrix[4*axis:4*axis+4,axis+2]=-.5*k[2]*b
        task.noise_matrix[4*axis:4*axis+4,2*axis]=b
        task.noise_matrix[4*axis:4*axis+4,2*axis+1]=g
    def action_bounds(lo,hi,el,eh):
        al,ah=_lin(-k[None],lo.reshape(2,4),hi.reshape(2,4));al,ah=al[:,0],ah[:,0]
        for i in range(2):
            l,h=_lin(np.array([[-k[0],-.5*k[2]]]),el[[i,i+2]],eh[[i,i+2]])
            al[i]+=l[0];ah[i]+=h[0]
        return al,ah
    def interval(lo,hi,el,eh,wl,wh,c,t):
        al,ah=action_bounds(lo,hi,el,eh)
        maximum=np.maximum(abs(al),abs(ah))
        if maximum.max()<=plant.AXIS_ACCEL and maximum.sum()<=plant.TOTAL_ACCEL:
            nl,nh=_lin(task.closed_matrix,lo,hi);e1,e2=_lin(task.error_matrix,el,eh);w1,w2=_lin(task.noise_matrix,wl,wh)
            return nl+e1+w1,nh+e2+w2
        # Genuine clipping and shared-current normalization: enclose the
        # positive common denominator, retaining all possible clipped actions.
        al,ah=np.clip(al,-plant.AXIS_ACCEL,plant.AXIS_ACCEL),np.clip(ah,-plant.AXIS_ACCEL,plant.AXIS_ACCEL)
        minimum,maximum=_abs(al,ah)
        denlo=max(1.,minimum.sum()/plant.TOTAL_ACCEL);denhi=max(1.,maximum.sum()/plant.TOTAL_ACCEL)
        al,ah=_mul((al,ah),(1/denhi,1/denlo))
        nl,nh=_lin(a,lo.reshape(2,4),hi.reshape(2,4))
        for j in range(2):
            dl,dh=_lin(b[:,None],np.array([al[j]+wl[2*j]]),np.array([ah[j]+wh[2*j]]))
            gl,gh=_lin(g[:,None],np.array([wl[2*j+1]]),np.array([wh[2*j+1]]))
            nl[j]+=dl+gl;nh[j]+=dh+gh
        return nl.ravel(),nh.ravel()
    rows=[];limits=[]
    for axis in range(2):
        for row,limit in [(np.eye(4)[i],v) for i,v in enumerate([1.45,1.8,.65,1.5])]+[(np.array([1,0,1,0]),1.55)]:
            full=np.zeros(8);full[4*axis:4*axis+4]=row;rows.append(full);limits.append(limit)
    rows.append(np.tile([1,0,1,0],2));limits.append(2.6)
    rows=np.asarray(rows);limits=np.asarray(limits)
    def safe(lo,hi,t):
        l,h=_lin(rows,lo,hi);a,z=_abs(l,h)
        return 'safe' if np.all(z<limits) else ('unsafe' if np.any(a>=limits) else 'unknown')
    @lru_cache(None)
    def facets(t):
        vectors=[];biases=[]
        for name,delay,row,offset,limit in certificate.planes(k,q,max(0,40-t),False):
            if name in ('total_accel','payload_diagonal'):
                signs=itertools.product([-1,1],repeat=2) if name=='total_accel' else [(-1,-1),(1,1)]
                for sx,sy in signs:
                    vectors.append(np.r_[sx*row,sy*row]/limit);biases.append(2*offset/limit)
            else:
                for axis,sign in itertools.product(range(2),[-1,1]):
                    full=np.zeros(8);full[4*axis:4*axis+4]=sign*row/limit;vectors.append(full);biases.append(offset/limit)
        return np.asarray(vectors),np.asarray(biases)
    def bb(lo,hi,t):
        mat,shift=facets(t);l,h=_lin(mat,lo,hi,shift)
        return np.maximum(0,l.max(1)),np.maximum(0,h.max(1))
    task._step=lambda x,e,w,c,t:plant.step(x.reshape(-1,2,4),plant.control(plant.observation(x.reshape(-1,2,4))+e,x.reshape(-1,2,4),k),w.reshape(-1,2,2)).reshape(-1,8)
    task._interval=interval;task._safe=safe
    task._barrier=lambda x,t:certificate.barrier(x.reshape(-1,2,4),k,q,max(0,40-t))
    task._barrier_bounds=bb;task._current_baseline=lambda:certificate.verify(k,q,exact=True)
    task._sample_initial=lambda n,seed:plant.scenarios(n,seed)[0].reshape(-1,8)
    def affine(lo,hi):
        al,ah=action_bounds(lo,hi,task.error_lo,task.error_hi)
        maximum=np.maximum(abs(al),abs(ah))
        if maximum.max()>plant.AXIS_ACCEL or maximum.sum()>plant.TOTAL_ACCEL:return None
        return task.closed_matrix,task.error_matrix,task.noise_matrix,np.zeros(8)
    task._affine=affine
    def affine_zonotope(center,generators):
        act=np.zeros((2,8));act[0,:4]=-k;act[1,4:]=-k
        obs=np.zeros((2,4))
        for i in range(2):obs[i,i]=-k[0];obs[i,i+2]=-.5*k[2]
        ac=act@center;ag=act@generators;eg=obs*q
        upper=np.abs(ac)+np.abs(ag).sum(axis=1)+np.abs(eg).sum(axis=1)
        if upper.max()>plant.AXIS_ACCEL:return None
        for sign in itertools.product([-1,1],repeat=2):
            sign=np.asarray(sign)
            if sign@ac+np.abs(sign@ag).sum()+np.abs(sign@eg).sum()>plant.TOTAL_ACCEL:return None
        return task.closed_matrix,task.error_matrix,task.noise_matrix,np.zeros(8)
    task._affine_zonotope=affine_zonotope
    task._safe_constraints=(np.r_[rows,-rows],np.r_[limits,limits])
    task.metadata.update(gains=k.tolist(),coordinates='axis x then y, each [p,v,s,r] physical SI units',
        initial_sampling='original task: |p| uniform .34..70, signs uniform; remaining components uniform',
        contract=_binding(folder/'contract.json'),controller=_binding(folder/'synthesis.json'),
        barrier_time_varying=True,unsafe_barrier_threshold=1.,current_baseline_rerun='exact rational finite-horizon support functions')
    return task


def _lane(results):
    from Aebs.lane_keeping import core
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.learned_sbc import load_sbc_checkpoint,bounds,transitions
    folder=results/'lane_robust_synthesis_20261116_v2'
    actor=load_actor(folder/'package/actor.pt').double().requires_grad_(False)
    barrier,_=load_sbc_checkpoint(folder/'package/sbc.pt');barrier=barrier.double().requires_grad_(False)
    q=_json(folder/'final_cp/calibration.json')['radius']
    task=_make('lane',2,2,core.HORIZON,core.DT,[-.25/.85,-.06/.30],[.25/.85,.06/.30],[-1,-1],[1,1],q,contexts=core.KAPPAS)
    def point(x,e,w,c,t):
        with _torch().no_grad():u=_np(actor(_tensor(np.c_[x+e,np.full(len(x),c/core.KAPPA_SCALE)]))).ravel()*core.PHI_MAX
        return core.dynamics_batch(x*[core.E_SCALE,core.THETA_SCALE],u,np.full(len(x),c))/[core.E_SCALE,core.THETA_SCALE]
    def interval(lo,hi,el,eh,wl,wh,c,t):
        with _torch().no_grad():l,h=transitions(actor,_tensor(lo)[None],_tensor(hi)[None],c,_tensor(el),_tensor(eh),'affine')
        return _np(l)[0],_np(h)[0]
    def bb(lo,hi,t):
        with _torch().no_grad():l,h=bounds(barrier,_tensor(lo),_tensor(hi),'affine')
        return _np(l),_np(h)
    def b(x,t):
        with _torch().no_grad():return _np(barrier(_tensor(x))).ravel()
    task._step=point;task._interval=interval;task._safe=lambda lo,hi,t:_rectangle_safe(lo,hi,task.domain_lo,task.domain_hi)
    task._barrier=b;task._barrier_bounds=bb
    task._torch_barrier_bounds=lambda a,b:bounds(barrier,a,b,'affine')
    def current(budget):
        from Aebs.lane_keeping.learned_sbc import error_boxes
        errors=error_boxes(q,4)
        def next_upper(a,b,context):
            following=_torch().zeros(len(a),dtype=a.dtype)
            for el,eh in errors:
                nl,nh=transitions(actor,a,b,context,el,eh,'affine')
                following=_torch().maximum(following,bounds(barrier,nl,nh,'affine')[1])
            return following
        regions={'initial':(task.initial_lo,task.initial_hi,True)}
        for j,(lo,hi) in enumerate([([-1.3,-1.3],[-1,1.3]),([1,-1.3],[1.3,1.3]),([-1,-1.3],[1,-1]),([-1,1],[1,1.3])]):
            regions['unsafe_'+str(j)]=(lo,hi,False)
        return _neural_current(task,barrier,[-1,-1],[1,1],regions,next_upper,budget)
    task._current_verify=current
    task._current_baseline=lambda:dict(status='archived_only',rerun=False,report=str(folder/'final_audit/audit.json'),
        reason='full adaptive continuous-domain neural audit must be run separately; not inferred from point samples')
    task.metadata.update(coordinates='normalized [e/.85,theta/.30]',contract=_binding(folder/'final_cp/calibration.json'),
         actor=_binding(folder/'package/actor.pt'),barrier=_binding(folder/'package/sbc.pt'),contexts=list(core.KAPPAS))
    return task


def _aebs(results):
    torch=_torch()
    from torch import nn
    from Combined_network.model import CombinedPolicyNetwork
    from Aebs.vision_sls.model import VisionSLSPolicy
    from Aebs.vision_sls.sbc_cp import load_sbc_checkpoint,monotone_box_bounds
    folder=results/'aebs_robust_synthesis_20261117_release'
    package=folder/'package'
    saved=torch.load(package/'policy.pt',map_location='cpu',weights_only=False)
    obs=torch.load(package/'observation.pt',map_location='cpu',weights_only=False)
    sizes=[2]+saved['actor_hidden'];layers=[]
    for a,b in zip(sizes[:-1],sizes[1:]):layers += [nn.Linear(a,b),nn.Tanh()]
    policy=VisionSLSPolicy(CombinedPolicyNetwork(nn.Sequential(*layers),nn.Linear(sizes[-1],1)),obs['distance_scale_m'])
    policy.load_state_dict(saved['state_dict']);policy=policy.double().requires_grad_(False)
    cfg=_json(package/'config.json');barrier,meta=load_sbc_checkpoint(package/'sbc.pt',cfg)
    barrier=barrier.double().requires_grad_(False);scale=float(meta.get('barrier_distance_scale_m',cfg['barrier_distance_scale_m']))
    audit=_json(folder/'final_audit/report.json');q=audit['envelope_radius_at_10m']
    task=_make('aebs',2,1,400,.05,[15,2.5],[16,3],[5,0],[16,3],q)
    seq=[*policy.controller_net.mlp_extractor,policy.controller_net.action_net]
    def terminal(x):
        return (x[...,0]<=6)|(x[...,0]>16)|(x[...,1]<=1e-6)
    def point(x,e,w,c,t):
        with torch.no_grad():u=_np(policy.forward_observation(_tensor((x[:,:1]+e)/policy.distance_scale_m),_tensor(x[:,1:]))).ravel()
        nxt=np.c_[x[:,0]-.05*x[:,1],np.clip(x[:,1]-.05*u,0,3)]
        return np.where(terminal(x)[:,None],x,nxt)
    def interval(lo,hi,el,eh,wl,wh,c,t):
        if hi[0]<=6 or lo[0]>16 or hi[1]<=1e-6:return lo.copy(),hi.copy()
        ol=np.array([(lo[0]+el[0])/policy.distance_scale_m,lo[1]])
        oh=np.array([(hi[0]+eh[0])/policy.distance_scale_m,hi[1]])
        al,ah=_network_bounds(seq,ol,oh);al,ah=np.clip(al,-3,3),np.clip(ah,-3,3)
        nl=np.array([lo[0]-.05*hi[1],np.clip(lo[1]-.05*ah[0],0,3)])
        nh=np.array([hi[0]-.05*lo[1],np.clip(hi[1]-.05*al[0],0,3)])
        if lo[0]<=6 or hi[0]>16 or lo[1]<=1e-6:nl,nh=np.minimum(nl,lo),np.maximum(nh,hi)
        return nl,nh
    def safe(lo,hi,t):
        if hi[0]<=6 and lo[1]>.5:return 'unsafe'
        if lo[0]<5 or hi[0]>16 or lo[1]<0 or hi[1]>3:return 'unknown'
        return 'safe' if lo[0]>6 or hi[1]<=.5 else 'unknown'
    def b(x,t):
        with torch.no_grad():return _np(barrier(_tensor(x/[scale,1]))).ravel()
    def bb(lo,hi,t):
        with torch.no_grad():l,h=monotone_box_bounds(barrier,_tensor(lo/[scale,1]),_tensor(hi/[scale,1]))
        return _np(l),_np(h)
    task._step=point;task._interval=interval;task._safe=safe;task._terminal=terminal
    task._barrier=b;task._barrier_bounds=bb
    task._torch_barrier_bounds=lambda a,b:monotone_box_bounds(barrier,a/_tensor([scale,1]),b/_tensor([scale,1]))
    def current(budget):
        from Aebs.vision_sls.verify_sbc_cells import state_boxes
        contract=_json(folder/'final_cp/perception_contract.json')
        def next_upper(a,b,context):
            return state_boxes(policy,barrier,scale,contract,a,b,actor_subdivisions=32,successor_subdivisions=16)[2]
        regions={'initial':([15,2.5],[16,3],True),'unsafe':([5,.5],[6,3],False)}
        return _neural_current(task,barrier,[6,1e-6],[16,3],regions,next_upper,budget)
    task._current_verify=current
    task._current_baseline=lambda:dict(status='archived_only',rerun=False,report=str(folder/'final_audit/report.json'),
        reason='782400-cell parent audit is archived, not rerun by this adapter')
    task.metadata.update(coordinates='distance m, velocity m/s',stopping='absorb success, unsafe, out of domain, or stopped-safe outside goal',
        actor=_binding(package/'policy.pt'),barrier=_binding(package/'sbc.pt'),
        error_source=_binding(folder/'final_audit/report.json'),barrier_distance_scale=scale,
        warning='q is the released constant tube; no new image calibration performed by adapter')
    return task


def _braking(results):
    from Aebs.curve_braking import plant
    from Aebs.curve_braking.models import Actor,Barrier
    from Aebs.curve_braking.verify import verify_checkpoint
    folder=results/'curve_braking_5d_20261117'
    actor=Actor().double();actor.load_state_dict(_torch().load(folder/'actor.pt',map_location='cpu',weights_only=True));actor.requires_grad_(False)
    barrier=Barrier().double();barrier.load_state_dict(_torch().load(folder/'barrier.pt',map_location='cpu',weights_only=True));barrier.requires_grad_(False)
    q=_json(folder/'evaluation/neural/contract.json')['radius']
    task=_make('braking',5,3,plant.HORIZON,plant.DT,plant.INIT_LO,plant.INIT_HI,
        [-.65,-.19,1.92,0,-plant.DMAX],[.65,.19,6.15,plant.VMAX,plant.DMAX],q,contexts=tuple(plant.KAPPAS))
    def point(x,e,w,c,t):
        with _torch().no_grad():u=_np(actor(_tensor(plant.normalize(x)+e),_tensor(x[:,3]),_tensor(x[:,4]),_tensor(np.full(len(x),c))))
        return plant.step(x,np.full(len(x),c),u)
    def interval(lo,hi,el,eh,wl,wh,c,t):
        if hi[3]<=0:return lo.copy(),hi.copy()
        il=np.r_[(lo[:3]-plant.CENTER)/plant.SCALE+el,lo[3]/2.6,lo[4]/.07,c/.02]
        ih=np.r_[(hi[:3]-plant.CENTER)/plant.SCALE+eh,hi[3]/2.6,hi[4]/.07,c/.02]
        a,b=_network_bounds(list(actor.net),il,ih)
        sig=lambda z:1/(1+np.exp(-np.clip(z,-700,700)))
        observed_lo,observed_hi=lo[2]+2*el[2],hi[2]+2*eh[2]
        floor_lo=max(-.4,1.2-.8*max(0,observed_hi-7.5));floor_hi=max(-.4,1.2-.8*max(0,observed_lo-7.5))
        brake_lo=floor_lo+(3.5-floor_lo)*sig(a[0]);brake_hi=floor_hi+(3.5-floor_hi)*sig(b[0])
        speed=(np.clip(lo[3]-.2*brake_hi,0,2.6),np.clip(hi[3]-.2*brake_lo,0,2.6));ds=(.2*speed[0],.2*speed[1])
        den=np.sort([1-c*lo[0],1-c*hi[0]])
        if den[0]<=0:return np.full(5,-np.inf),np.full(5,np.inf)
        adv=_mul(_mul(ds,_trig(lo[1],hi[1],True)),(1/den[1],1/den[0]))
        lateral=_mul(ds,_trig(lo[1],hi[1]));yaw=_mul(ds,(np.tan(lo[4])/2.6,np.tan(hi[4])/2.6))
        curve=_mul((c,c),adv);steer=_mul(ds,(.16*np.tanh(a[1]),.16*np.tanh(b[1])))
        nl=np.array([lo[0]+lateral[0],lo[1]+yaw[0]-curve[1],lo[2]-adv[1],speed[0],np.clip(lo[4]+steer[0],-.07,.07)])
        nh=np.array([hi[0]+lateral[1],hi[1]+yaw[1]-curve[0],hi[2]-adv[0],speed[1],np.clip(hi[4]+steer[1],-.07,.07)])
        if lo[3]<=0:nl,nh=np.minimum(nl,lo),np.maximum(nh,hi)
        return nl,nh
    def safe(lo,hi,t):
        ae,be=_abs(lo[0],hi[0]);at,bt=_abs(*_trig(lo[1],hi[1]))
        rho=np.hypot(plant.HALF_LENGTH,plant.HALF_WIDTH);k=max(abs(plant.KAPPAS))
        denlo=1-k*(be+rho);denhi=1-k*(ae+rho)
        if denlo<=0:return 'unknown'
        lane_lo=plant.ROAD_HALF-be-plant.HALF_WIDTH-plant.HALF_LENGTH*bt-.5*k*rho*rho/denlo
        lane_hi=plant.ROAD_HALF-ae-plant.HALF_WIDTH-plant.HALF_LENGTH*at-.5*k*rho*rho/denhi
        gaplo,gaphi=lo[2]-rho/denlo,hi[2]-rho/denhi
        if lane_lo>0 and gaplo>0 and lo[3]>=0 and hi[3]<=2.6 and lo[4]>=-.07 and hi[4]<=.07:return 'safe'
        return 'unsafe' if lane_hi<=0 or gaphi<=0 else 'unknown'
    coeff=_np(barrier.coefficients());knots=_np(barrier.knots)
    def bb(lo,hi,t):
        ae,at,beta,ad=coeff
        el,eh=_abs(lo[:,0],hi[:,0]);pl,ph=_abs(lo[:,1],hi[:,1]);dl,dh=_abs(lo[:,4],hi[:,4])
        rl=.2*np.maximum(0,lo[:,3,None]-knots).sum(1);rh=.2*np.maximum(0,hi[:,3,None]-knots).sum(1)
        l=np.c_[(el+ae*rl-.65)/.028,(pl+beta*dl+at*rl-.19)/.014,(1.92+ad*rl-hi[:,2])/.18,(lo[:,2]-6.15)/.15]
        h=np.c_[(eh+ae*rh-.65)/.028,(ph+beta*dh+at*rh-.19)/.014,(1.92+ad*rh-lo[:,2])/.18,(hi[:,2]-6.15)/.15]
        return 20*np.maximum(0,1+l.max(1)),20*np.maximum(0,1+h.max(1))
    def b(x,t):
        with _torch().no_grad():return _np(barrier(_tensor(x)))
    task._step=point;task._interval=interval;task._safe=safe;task._barrier=b;task._barrier_bounds=bb
    task._terminal=lambda x:x[...,3]<=0
    task._current_baseline=lambda:verify_checkpoint(folder/'barrier.pt',q)
    task.metadata.update(coordinates='physical [e,heading,distance,speed,wheel_angle]',
        actor=_binding(folder/'actor.pt'),barrier=_binding(folder/'barrier.pt'),
        contract=_binding(folder/'evaluation/neural/contract.json'),current_baseline_rerun='global structural scalar inequalities',
        barrier_separation_domain='0<=v<=2.6 and |wheel_angle|<=.07; B<20 geometric sublevel')
    return task


def load_task(name, root=None, variant=None):
    """root is the artical-F122 directory (or its results directory).

    Only the frozen uniform contract variant is implemented. Rejecting other
    values prevents accidentally labelling a box outer approximation as the
    original action-coupled or state-dependent set.
    """
    if variant not in (None,'uniform'):
        raise ValueError('only registered uniform residual contracts supported')
    if name not in NAMES:
        raise ValueError(f'unknown task {name!r}; choose {NAMES}')
    results=RESULTS if root is None else Path(root)
    if results.name!='results':results=results/'results'
    task=globals()['_'+name](results)
    task.metadata.update(adapter=_binding(Path(__file__)),task=name,horizon=task.horizon,dt=task.dt,
        error_lo=task.error_lo.tolist(),error_hi=task.error_hi.tolist(),noise_lo=task.noise_lo.tolist(),noise_hi=task.noise_hi.tolist(),
        original_cp_distribution_unchanged=False,scope='conditional on frozen residual support; backend sampling is not new image CP')
    return task
