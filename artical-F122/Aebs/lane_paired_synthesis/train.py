"""Matched warm-start actor/barrier resynthesis before fresh image calibration."""
import argparse
import copy
import json
import signal
import time
from contextlib import contextmanager
from itertools import product
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import (grid_boxes, load_sbc_checkpoint, raw, save_actor,
                                       save_barrier, sha256, unsafe_boxes)
from ..lane_keeping.policy import load_actor
from ..lane_contract_comparison.audit import find_witness
from ..lane_keeping.robust_trajectory_cp import write_json


ROOT = Path('results/lane_paired_synthesis_20261117_v1')
BASE = Path('results/lane_robust_synthesis_20261116_v2/package')
DEVELOPMENT = Path('results/lane_contract_comparison_20261117_v1')
ARMS = ('uniform','diagonal','joint')
CURVES = (-.02,-.01,0.,.01,.02)
EXTRA = (0.,.1,.15,.19,.2)  # normalized steering
STEPS = 80
ROUNDS = 3
SEED = 202611178300
PAST_EXTRA = .20
FINAL_AUDIT_SECS = 180


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError('registered wall-clock budget')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def read(path):
    return json.loads(path.read_text())


def register(root):
    root.mkdir(parents=True,exist_ok=False)
    inputs=[BASE/'actor.pt',BASE/'sbc.pt',BASE/'patch_projection.pt',DEVELOPMENT/'contracts.json',
            DEVELOPMENT/'protocol.json',Path(__file__),Path(__file__).with_name('audit.py')]
    dev=read(DEVELOPMENT/'contracts.json')
    assert dev['protocol']==sha256(DEVELOPMENT/'protocol.json')
    base_actor=load_actor(BASE/'actor.pt')
    base_barrier=load_sbc_checkpoint(BASE/'sbc.pt')[0]
    protocol=dict(arms=ARMS,kappas=CURVES,alpha=.05,visual_weight=.5,horizon=80,
        initialization='identical warm start from frozen robust v2 actor and B for all arms, last iterate; no selection',
        model_num_parameters={n:sum(p.numel() for p in model.parameters()) for n,model in
                              [('actor',base_actor),('barrier',base_barrier)]},
        actor_layers=[(a.in_features,a.out_features) for a in base_actor.modules() if isinstance(a,torch.nn.Linear)],
        barrier_layers=[(a.in_features,a.out_features) for a in base_barrier.modules() if isinstance(a,torch.nn.Linear)],
        inputs_sha256={str(p):sha256(p) for p in inputs}, development_contracts=dev['arms'],
        development_scale=read(DEVELOPMENT/'protocol.json')['development']['scale'],
        seeds=dict(training=SEED,calibration=202611178301,holdout=202611178302),n_cal=199,n_holdout=100,
        training=dict(cegis_rounds=ROUNDS,actor_steps_per_round=STEPS,barrier_steps_per_round=STEPS,
                      initial_networks_identical=True,cegis_search_seconds=30,cegis_grid=52,cegis_extra=PAST_EXTRA,
                      actor_lr=3e-5,barrier_lr=2e-4,threads=1,batch=96,unroll=12,
                      feedback='actual positive-drift witness from each curvature; no replacement of timeouts by passes',
                      objective='12-step nominal/adversarial tracking, worst candidate point dynamics and regional B loss'),
        final_audit=dict(grid=40,max_depth=16,error_subdivisions=2,max_pending=30000,
                         seconds_per_margin_per_arm=FINAL_AUDIT_SECS,
                         margins=EXTRA,curvatures=CURVES,
                         region_thresholds=[1,20], stop='time or unresolved recorded; no safety inference'),
        outcomes='same paired initial and curvature draws across arms, fresh rendered 80 steps at each visited state; report all statuses',
        scope='registered five discrete constant curvatures, no continuous-curvature extrapolation')
    write_json(root/'protocol.json',protocol)


def control(actor,x,k,err=None):
    obs=x if err is None else x+err
    return actor(torch.cat((obs,(k/core.KAPPA_SCALE)[:,None]),dim=1)).flatten()


def successor(x,k,u):
    e=x[...,0]+core.SPEED*core.DT/core.E_SCALE*torch.sin(x[...,1]*core.THETA_SCALE)
    theta=x[...,1]+core.SPEED*core.DT/core.THETA_SCALE*(torch.tan(u*core.PHI_MAX)/core.WHEELBASE-k)
    return torch.stack((e,theta),dim=-1)


def within_cap(action,nominal,cap):
    if cap is None:
        return action
    return nominal+(action-nominal).clamp(-cap,cap)


def curve_draw(n):
    return torch.tensor(CURVES)[torch.randint(len(CURVES),(n,))]


def rollout_loss(actor,width,cap):
    n=96
    x=(2*torch.rand(n,2)-1)*torch.tensor([core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE])
    k=curve_draw(n)
    # Share a state, curvature and direction draw across the three stress scenarios.
    bias=(2*torch.rand(n,2)-1)*width
    x=x.repeat(3,1)
    k=k.repeat(3)
    error=torch.cat([torch.zeros_like(bias),bias,-bias])
    obj=torch.zeros_like(k)
    last=None
    for step in range(12):
        nominal=control(actor,x,k)
        cmd=within_cap(control(actor,x,k,error),nominal,cap)
        x=successor(x,k,cmd)
        obj+=x[:,0].square()+.20*x[:,1].square()+.004*cmd.square()
        if last is not None:
            obj+=.01*(cmd-last).square()
        last=cmd
    scenarios=obj.reshape(3,n)/12
    return (.5*scenarios[0]+.5*scenarios.max(0).values).mean()


def safety_loss(actor,barrier,width,cap,feedback):
    n=96
    x=2*torch.rand(n,2)-1
    k=curve_draw(n)
    witness_uniform=torch.rand(24)
    witness_jitter=torch.randn(24,2)
    if feedback:
        data=torch.tensor([[*r['x'],r['kappa']] for r in feedback])
        ids=(witness_uniform*len(data)).long()
        x[:24]=(data[ids,:2]+.02*witness_jitter).clamp(-1,1)
        x[:len(feedback)]=data[:len(feedback),:2]
        k[:24]=data[ids,2]
        k[:len(feedback)]=data[:len(feedback),2]
    mesh=x.new_tensor(list(product((-1.,0.,1.),repeat=2)))*width
    observed=control(actor,(x[:,None]+mesh).reshape(-1,2),k.repeat_interleave(9)).reshape(n,9)
    nominal=control(actor,x,k)[:,None]
    observed=within_cap(observed,nominal,cap)
    command=torch.stack([(observed.min(1).values-PAST_EXTRA).clamp(-1,1),
                         (observed.max(1).values+PAST_EXTRA).clamp(-1,1)],1)
    expanded=x[:,None,:].expand(-1,2,-1)
    after=raw(barrier,successor(expanded,k[:,None],command).reshape(-1,2)).reshape(n,2).max(1).values
    before=barrier(x).flatten()
    # A 0.1 raw-output margin keeps the ReLU-zero plateau trainable.
    drift=(after-before+.1).relu().masked_fill(before.detach()>20,0.)
    return (drift.mean()+drift.max())/20


def region_loss(barrier):
    init=(2*torch.rand(120,2)-1)*torch.tensor([core.INIT_E/core.E_SCALE,core.INIT_THETA/core.THETA_SCALE])
    low,high=unsafe_boxes(4,dtype=torch.float32)
    ids=torch.randint(len(low),(128,))
    unsafe=low[ids]+(high[ids]-low[ids])*torch.rand(128,2)
    init_error=(raw(barrier,init)+.1).relu().max()
    unsafe_error=(25-raw(barrier,unsafe)).relu().max()
    return (init_error+unsafe_error)/20


def witness_search(actor,barrier,contract):
    lo,hi=grid_boxes([-1.,-1.],[1.,1.],52)
    points=(lo+hi)/2
    a=copy.deepcopy(actor).double().requires_grad_(False)
    b=copy.deepcopy(barrier).double().requires_grad_(False)
    found=[]
    status='completed'
    completed=[]
    start=time.monotonic()
    with torch.no_grad():
        try:
            with deadline(30):
                for k in CURVES:
                    candidate=find_witness(a,b,points,k,contract,PAST_EXTRA)
                    if candidate:
                        found.append(candidate)
                    completed.append(k)
        except TimeoutError:
            status='timeout'
    return dict(status=status,completed_curvatures=completed,witnesses=found,seconds=time.monotonic()-start)


def update(opt,loss,model):
    if not torch.isfinite(loss):
        raise ValueError('invalid training loss')
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    opt.step()


def train(root,name):
    protocol=read(root/'protocol.json')
    if protocol['inputs_sha256'][str(Path(__file__))]!=sha256(Path(__file__)):
        raise ValueError('training source changed')
    out=root/name
    out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    actor=load_actor(BASE/'actor.pt')
    barrier=load_sbc_checkpoint(BASE/'sbc.pt')[0]
    actopt=torch.optim.Adam(actor.parameters(),lr=protocol['training']['actor_lr'])
    baropt=torch.optim.Adam(barrier.parameters(),lr=protocol['training']['barrier_lr'])
    contract=protocol['development_contracts'][name]
    width=torch.tensor(contract['width'],dtype=torch.float32)
    cap=contract['action']
    history=[]
    feedback=[]
    for round_id in range(ROUNDS):
        search=witness_search(actor,barrier,contract)
        feedback.extend(search['witnesses'])
        # All arms draw the same ordinary states even if different witnesses were found.
        for phase in ('actor','barrier'):
            torch.manual_seed(SEED+10*round_id+(phase=='barrier'))
            actor.requires_grad_(phase=='actor')
            barrier.requires_grad_(phase=='barrier')
            for _ in range(STEPS):
                loss=safety_loss(actor,barrier,width,cap,feedback)
                if phase=='actor':
                    loss=rollout_loss(actor,width,cap)+.4*loss
                    update(actopt,loss,actor)
                else:
                    loss=region_loss(barrier)+loss
                    update(baropt,loss,barrier)
        history.append(dict(round=round_id+1,feedback=search,training_witnesses=len(feedback),
                            completed_actor_steps=STEPS,completed_barrier_steps=STEPS))
        write_json(out/'training.json',history)
        print(name,'round',round_id+1,'witnesses',len(feedback),flush=True)
    save_actor(out/'actor.pt',actor)
    save_barrier(out/'sbc.pt',barrier,dict(training='paired warm-start resynthesis',arm=name))
    changed={}
    for key,model,orig in [('actor',actor,load_actor(BASE/'actor.pt')),
                           ('barrier',barrier,load_sbc_checkpoint(BASE/'sbc.pt')[0])]:
        diff=sum((p-q).square().sum() for p,q in zip(model.parameters(),orig.parameters()))
        changed[key]=float(diff.detach().sqrt())
        if changed[key]<=0:raise ValueError(f'{key} did not change')
    write_json(out/'frozen.json',dict(protocol=sha256(root/'protocol.json'),arm=name,
               actor_sha256=sha256(out/'actor.pt'),sbc_sha256=sha256(out/'sbc.pt'),
               training_history=sha256(out/'training.json'),weights_l2_change=changed,
               completed_updates_per_network=ROUNDS*STEPS))
    print(name,'frozen',changed,flush=True)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('register',*ARMS))
    cli.add_argument('--root',type=Path,default=ROOT)
    opt=cli.parse_args()
    if opt.stage=='register':
        register(opt.root)
    else:
        train(opt.root,opt.stage)
