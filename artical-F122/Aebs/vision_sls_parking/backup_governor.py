"""Belief-preserving action governor, with no true state/scenario input.

Delayed measurements prune hypotheses only under registered CP compatibility.
Boxes crossing the CP domain boundary are retained without measurement pruning.
"""
import itertools
import numpy as np
from mpmath import iv
from .core import INITIAL_STATES,C,RENDER_OFFSET,ACTION_HIGH,GOAL_LOW,GOAL_HIGH,nominal_observation,dynamics
from .planning_feedback import PlannedController
from .perception_error import radius
from .interval_backup import point,add,mul,step,safe,backup,braking_action,sequence_safe,sequence_box


def canonical_action(action):
    value=np.clip(np.asarray(action,dtype=float),-ACTION_HIGH,ACTION_HIGH)
    for _ in range(16):
        nxt=np.clip(value/ACTION_HIGH,-1,1)*ACTION_HIGH
        if np.array_equal(nxt,value):return value
        value=nxt
    raise RuntimeError('action normalization did not reach a fixed point')


def radius_upper(vertices,model):
    """Validated vertex maxima of the registered multi-affine radius."""
    result=np.zeros(3)
    bits=list(itertools.product([False,True],repeat=3))
    for x in vertices:
        t=[(iv.mpf(float(x[k]))-iv.mpf(float(model['low'][k])))/
           (iv.mpf(float(model['high'][k]))-iv.mpf(float(model['low'][k]))) for k in range(3)]
        for output in range(3):
            value=iv.mpf(0)
            for row,b in enumerate(bits):
                weight=iv.mpf(float(model['coefficients'][row][output]))
                for k in range(3):weight*=t[k] if b[k] else 1-t[k]
                value+=weight
            value*=iv.mpf(float(model['q']))
            result[output]=max(result[output],np.nextafter(float(value.b),np.inf))
    return result


def compatible(box,y,error_model):
    keep=[]
    for low,high in zip(*box):
        if (low[:3]<error_model['low']).any() or (high[:3]>error_model['high']).any():
            keep.append(True);continue
        # Multi-affine Bernstein radii attain each coordinate maximum at a vertex.
        vertices=np.asarray([np.where(bits,high[:3],low[:3]) for bits in itertools.product([False,True],repeat=3)])
        r=radius_upper(vertices,error_model)
        hlow=np.zeros(3);hhigh=np.zeros(3)
        translated=add((low,high),point(RENDER_OFFSET))
        for k in range(4):
            term=mul((translated[0][k],translated[1][k]),point(C[:,k]))
            hlow,hhigh=add((hlow,hhigh),term)
        hlow,hhigh=add((hlow,hhigh),(-r,r))
        keep.append(bool(((y>=hlow)&(y<=hhigh)).all()))
    return np.asarray(keep)


def initial_belief(y,goal,error_model,jitter=0.):
    if not np.isfinite(jitter) or jitter<0:
        raise ValueError('finite nonnegative initial box radius required')
    centers=(GOAL_LOW[:,:2]+GOAL_HIGH[:,:2])/2
    ids=np.flatnonzero(np.linalg.norm(centers-goal,axis=1)<1e-8)
    boxes=add(point(INITIAL_STATES[ids]),(-jitter,jitter))
    if error_model.get('schema')=='parking_startup_group_cp_v1':
        if jitter>error_model['initial_bound']:raise ValueError('initial box exceeds startup calibration domain')
        keep=[]
        for row,j in enumerate(ids):
            translated=add((boxes[0][row],boxes[1][row]),point(RENDER_OFFSET))
            h=point(np.asarray(error_model['bias'][j]))
            for k in range(4):h=add(h,mul((translated[0][k],translated[1][k]),point(C[:,k])))
            r=np.asarray(error_model['radius'][j]);h=add(h,(-r,r))
            keep.append(bool(((y>=h[0])&(y<=h[1])).all()))
        candidates=ids[np.asarray(keep)]
    else:candidates=ids[compatible(boxes,y,error_model)]
    if not len(candidates):
        raise ValueError('initial measurement incompatible with all registered hypotheses')
    return np.asarray(candidates), add(point(INITIAL_STATES[candidates]),(-jitter,jitter))


class BackupGovernor:
    def __init__(self,bank,policy,error_model,initial_jitter=0.,allowance=1e-12,library=None,startup_model=None):
        self.controller=PlannedController(bank,policy,mode='residual')
        self.error_model=error_model;self.initial_jitter=initial_jitter;self.allowance=allowance
        self.library=library
        self.startup_model=startup_model
        if startup_model is not None:
            if startup_model.get('schema')!='parking_startup_group_cp_v1':
                raise ValueError('unsupported startup calibration schema')
            for key in ['bias','radius']:
                array=np.asarray(startup_model[key],dtype=float)
                if array.shape!=(15,3) or not np.isfinite(array).all():
                    raise ValueError('finite 15 by 3 startup '+key+' required')
            if (np.asarray(startup_model['radius'])<=0).any():raise ValueError('positive startup radii required')
            if 'initial_states' in startup_model and not np.array_equal(startup_model['initial_states'],INITIAL_STATES):
                raise ValueError('startup initial catalog mismatch')
            if not 0<=initial_jitter<=startup_model['initial_bound']:raise ValueError('startup domain mismatch')

    def reset(self,y,goal):
        self.hypotheses,self.box=initial_belief(np.asarray(y),np.asarray(goal),
                    self.error_model if self.startup_model is None else self.startup_model,self.initial_jitter)
        self.controller.reset(y,goal);self.time=0;self.interventions=0;self.minimum_blend=1.
        self.invalid_measurements=0;self.invalid_proposals=0
        if self.startup_model is not None:
            # Select using only compatible visual hypotheses, never the true ID.
            costs=[]
            for j in self.hypotheses:
                bias=np.asarray(self.startup_model['bias'][j]);scale=np.maximum(self.startup_model['radius'][j],1e-8)
                costs.append(np.sum(((y-bias-nominal_observation(INITIAL_STATES[j]))/scale)**2))
            j=int(self.hypotheses[int(np.argmin(costs))]);prior=self.controller.bank.states[j,0]
            self.controller.reference=j
            corrected=np.asarray(y)-np.asarray(self.startup_model['bias'][j])
            self.controller.estimate=prior+self.controller.bank.l[j,0]@(corrected-nominal_observation(prior))
        self.backup_actions=None;self.backup_reference=None;self.goal_backup_certified=False
        self.goal_index=(int(np.flatnonzero(self.hypotheses==self.controller.reference)[0])
                         if self.controller.reference in self.hypotheses else None)
        candidates=[self.controller.reference]+[j for j in range(15) if j!=self.controller.reference and
                    np.linalg.norm(self.controller.bank.goals[j]-goal)<1e-8]
        sequences=[]
        if self.library is not None:
            for i,(j,mask) in enumerate(zip(self.library['references'],self.library['masks'])):
                if j==self.controller.reference and np.array_equal(np.flatnonzero(mask),self.hypotheses):
                    sequences.append(('library:'+str(i),self.library['actions'][i]))
        sequences.extend((j,self.controller.bank.actions[j]) for j in candidates)
        safe_fallback=None
        for j,raw in sequences:
            actions=np.asarray([canonical_action(u) for u in raw])
            terminal=sequence_box(self.box,actions,self.allowance)
            if terminal is not None:
                if safe_fallback is None:safe_fallback=(actions,j)
                if self._goal_contains(terminal):
                    self.backup_actions=actions;self.backup_reference=j;self.goal_backup_certified=True;break
        if self.backup_actions is None and safe_fallback is not None:
            self.backup_actions,self.backup_reference=safe_fallback
        if self.backup_actions is None and not backup(self.box,0,self.allowance):
            raise ValueError('initial belief has no certified reference or braking backup')
        return self.controller.features()

    def _tail_safe(self,box,time):
        if self.backup_actions is None:return backup(box,time,self.allowance)
        terminal=sequence_box(box,self.backup_actions[time:],self.allowance)
        return terminal is not None and (not self.goal_backup_certified or self._goal_contains(terminal))

    def _goal_contains(self,box):
        if self.goal_index is None:return False
        j=self.controller.reference;i=self.goal_index
        return bool((box[0][i]>=GOAL_LOW[j]-.001).all() and (box[1][i]<=GOAL_HIGH[j]+.001).all())

    def action(self,residual=None):
        if self.time>=30:raise RuntimeError('horizon complete')
        emergency=braking_action(self.box) if self.backup_actions is None else self.backup_actions[self.time]
        try:
            with np.errstate(over='raise',invalid='raise'):
                proposal=self.controller.action(residual)
            if not np.isfinite(proposal).all():raise ValueError('invalid proposal')
        except (ValueError,FloatingPointError):
            proposal=emergency.copy();self.invalid_proposals+=1
        for fraction in [1.,.75,.5,.25,0.]:
            u=canonical_action(emergency+fraction*(proposal-emergency))
            nxt=step(self.box,u,self.allowance)
            if safe(nxt) and self._tail_safe(nxt,self.time+1):
                self.capture_box=(self.box[0].copy(),self.box[1].copy())
                self.box=nxt;self.controller.last_action=u.copy()
                self.interventions+=int(fraction<1);self.minimum_blend=min(self.minimum_blend,fraction)
                return u
        raise RuntimeError('backup invariant lost; no unchecked action is executed')

    def receive(self,y):
        old=self.controller.estimate.copy();self.time+=1
        y=np.asarray(y)
        if y.shape==(3,) and np.isfinite(y).all() and np.max(np.abs(y))<1e6:
            self.controller.receive(y)
        else:
            self.invalid_measurements+=1;self.controller.t=self.time
            self.controller.estimate=dynamics(old,self.controller.last_action)
        # With startup calibration we never prune later: subsequent vision
        # errors can be arbitrary without invalidating the safety belief.
        keep=(compatible(self.capture_box,np.asarray(y),self.error_model) if self.startup_model is None and y.shape==(3,) and np.isfinite(y).all() else
              np.ones(len(self.hypotheses),dtype=bool))
        if keep.any() and not keep.all():
            self.hypotheses=self.hypotheses[keep];self.box=(self.box[0][keep],self.box[1][keep])
            self.goal_index=(int(np.flatnonzero(self.hypotheses==self.controller.reference)[0])
                             if self.controller.reference in self.hypotheses else None)
            if self.goal_index is None:self.goal_backup_certified=False
            if self.time<30:
                candidates=[self.controller.reference]+[i for i in range(15) if i!=self.controller.reference and
                    np.linalg.norm(self.controller.bank.goals[i]-self.controller.goal)<1e-8]
                for j in candidates:
                    actions=np.asarray([canonical_action(u) for u in self.controller.bank.actions[j]])
                    end=sequence_box(self.box,actions[self.time:],self.allowance)
                    if end is not None and self._goal_contains(end):
                        self.backup_actions=actions;self.backup_reference=j;self.goal_backup_certified=True;break
        # Known initial speed interval + executed accelerations supplies speed.
        # No simulator speed enters either the observer or the action proposal.
        self.controller.estimate[3]=(self.box[0][:,3].min()+self.box[1][:,3].max())/2
        return self.controller.features()
