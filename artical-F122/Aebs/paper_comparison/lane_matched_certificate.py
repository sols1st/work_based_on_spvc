"""Evaluate each retrained lane system's own safety interface."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .lane_matched_data import write_json
from .lane_matched_native import sha


def read(path):
    return json.loads(path.read_text())


def our_run(output, actor, radius, phase, *, barrier=None, epochs=4000, resume=None, extra=None, seed=2026112654):
    command = [sys.executable, '-u', '-m', 'Aebs.paper_comparison.spvt_task_own_barrier', phase,
               '--actor', str(actor), '--output', str(output), '--radius', str(radius),
               '--steering-limit', str(math.pi/6), '--bound-method', 'affine',
               '--depth', '19', '--epochs', str(epochs), '--seed', str(seed)]
    for key, path in [('--barrier', barrier), ('--resume',resume), ('--extra-states',extra)]:
        if path:
            command.extend([key,str(path)])
    with (output.parent / (output.name+'.log')).open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def own_train(experiment):
    output = experiment/'own_certificate'
    output.mkdir(exist_ok=False)
    actor = experiment/'own/actor.pt'
    train_radius = read(experiment/'own/training.json')['design_radius']
    candidates = []
    for turn in range(3):
        training = output/f'train_{turn}'
        audit = output/f'development_{turn}'
        prev = output/f'train_{turn-1}'/'sbc.pt'
        ce = output/f'development_{turn-1}'/'feedback.npz'
        our_run(training, actor, train_radius, 'train', resume=prev if prev.exists() else None,
                extra=ce if ce.exists() else None, epochs=5000 if turn else 4000)
        barrier = training/'sbc.pt'
        our_run(audit, actor, train_radius, 'audit', barrier=barrier)
        outcome = read(audit/'certificate.json')
        candidates.append(dict(turn=turn, barrier=str(barrier), sha256=sha(barrier),
                               passed=outcome['passed'], report=str(audit/'certificate.json')))
        if outcome['passed']:
            break
    write_json(output/'development.json', dict(radius=train_radius,
        actor_sha256=sha(actor), candidates=candidates, selected=candidates[-1]))


def own_audit(experiment):
    output = experiment/'own_certificate'
    actor = experiment/'own/actor.pt'
    data = read(output/'development.json')
    barrier = Path(data['selected']['barrier'])
    assert sha(barrier)==data['selected']['sha256'] and sha(actor)==data['actor_sha256']
    cal = experiment/'evaluation_own/own_thresholds.json'
    q = read(cal)['observation_score']['q']
    our_run(output/'final',actor,q,'audit',barrier=barrier)
    write_json(output/'summary.json', dict(radius=q, passed=read(output/'final/certificate.json')['passed'],
               barrier_sha256=sha(barrier), calibration_sha256=sha(cal)))


def native(experiment):
    root = (experiment/'native').resolve()
    sys.path[:0] = [str(root),str(root/'spvt')]
    from model.model import get_model
    from spvt.utils import create_bounded_module
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm

    output = experiment/'native_certificate'
    output.mkdir(exist_ok=False)
    selection = read(experiment/'evaluation_native/native_selection.json')['selected']
    checkpoint = Path(selection['path'])
    assert sha(checkpoint)==selection['sha256']
    model = get_model()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict'])
    network = create_bounded_module(argparse.Namespace(device='cpu'),model.eval())
    network.eval()

    @torch.no_grad()
    def bound(low, high):
        x = BoundedTensor((low+high)/2, PerturbationLpNorm(norm=np.inf,x_L=low,x_U=high))
        z = torch.zeros(len(low),10)
        latent = BoundedTensor(z, PerturbationLpNorm(norm=np.inf,x_L=z-.8,x_U=z+.8))
        l, u = network.compute_bounds(x=(latent,x), method='IBP')
        lower, upper = l.flatten().clamp(-1,1), u.flatten().clamp(-1,1)
        return lower,upper

    archive = dict(np.load(experiment/'evaluation_native/spvt_test.npz'))
    initial = torch.from_numpy(archive['states'][:,0].copy())
    prefix = np.zeros(len(initial),dtype=int)
    first_contains = []
    for j in range(0,len(initial),64):
        low = high = initial[j:j+64]
        ids = torch.arange(j,j+len(low))
        for step in range(160):
            action_lo,action_hi=bound(low,high)
            if step==0:
                actual=torch.from_numpy(archive['actions'][ids,0])
                first_contains.extend(((action_lo-1e-6<=actual)&(actual<=action_hi+1e-6)).tolist())
            # Restrict heading to the monotone sine region. Resolve only
            # boxes that remain in this region and in the lateral safety band.
            def propagate(x,u):
                angle = torch.deg2rad(x[:,1])
                steering = torch.deg2rad(u*30)
                return torch.stack([x[:,0]+25*torch.sin(angle),
                                    x[:,1]+torch.rad2deg(.1*torch.tan(steering))],1)
            low,high=propagate(low,action_lo),propagate(high,action_hi)
            valid=(low[:,0]>=-80)&(high[:,0]<=80)&(low[:,1]>=-90)&(high[:,1]<=90)
            ids,low,high=ids[valid],low[valid],high[valid]
            prefix[ids.numpy()]=step+1
            if not len(ids):
                break
    np.savez_compressed(output/'reachable_prefix.npz', prefixes=prefix, first_contains=first_contains)
    write_json(output/'summary.json', dict(seeds_checked=len(prefix), certified_160=int((prefix==160).sum()),
        certified_10=int((prefix>=10).sum()), prefix_min=int(prefix.min()), prefix_max=int(prefix.max()),
        prefix_median=float(np.median(prefix)),
        camera_first_action_in_proxy_interval=sum(first_contains),
        camera_first_excluded=len(first_contains)-sum(first_contains),
        model_sha256=sha(checkpoint), full_latent_box=[-.8,.8],
        method='original native SPVT combined model with LiRPA IBP at each step, state interval from native dynamics',
        arithmetic='numeric, without directed rounding',
        scope='original learned latent proxy for this newly trained controller; first-frame camera inclusion is a separate diagnostic'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('phase',choices=['own_train','own_audit','native'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    {'own_train':own_train,'own_audit':own_audit,'native':native}[args.phase](args.experiment)
