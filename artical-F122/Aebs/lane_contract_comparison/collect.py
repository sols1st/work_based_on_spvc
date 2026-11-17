"""Calibrate three sets on shared new image trajectories, after freezing all scoring rules."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.renderer import RoadRenderer
from ..lane_keeping.robust_trajectory_cp import DevicePatchObserver,collect,make_protocol,write_json

ROOT=Path('results/lane_robust_synthesis_20261116_v2')
VISUAL_WEIGHT=.5
MARGINS=[0.,.05,.1,.15,.2,.25,.30]
ARMS=('uniform','diagonal','joint')


@torch.inference_mode()
def maxima(data, actor):
    errors=np.asarray(data['errors'])
    x=data['states'][:,:-1]/[core.E_SCALE,core.THETA_SCALE]
    curve=np.broadcast_to(data['kappas'][:,None,None]/core.KAPPA_SCALE,(*x.shape[:2],1))
    def actions(position):
        z=torch.from_numpy(np.concatenate([position,curve],axis=-1)).double().reshape(-1,3)
        return torch.cat([actor(row) for row in z.split(2048)]).numpy().reshape(x.shape[:2])
    # Evaluate both clipped neural actions for the mathematical policy. Raw
    # rollouts also store actually applied actions for a precision crosscheck.
    actual=actions(data['observations'])
    if np.max(np.abs(data['actions']-actual))>2e-6:
        raise ValueError('rollout actor disagrees with registered weights')
    return np.abs(errors).max(axis=1),np.abs(actual-actions(x)).max(axis=1)


def scores(extrema,scale):
    visual,act=extrema
    diagonal=np.max(visual/scale['visual'],axis=1)
    return dict(uniform=visual.max(axis=1),diagonal=diagonal,
                joint=np.maximum(VISUAL_WEIGHT*diagonal,act/scale['action']))


def parameters(name,q,scale):
    if name=='uniform':
        return dict(width=[q,q],action=None)
    divisor=VISUAL_WEIGHT if name=='joint' else 1
    return dict(width=(np.array(scale['visual'])*q/divisor).tolist(),
                action=q*scale['action'] if name=='joint' else None)


class DoubleAction:
    def __init__(self,actor):
        self.actor=actor

    def __call__(self,obs):
        return self.actor(obs.double())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('output_dir',type=Path)
    parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args()
    args.package=ROOT/'package'
    args.dino_repo=Path('../external/dinov2')
    args.dino_weights=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth')
    args.calibration_seed,args.holdout_seed=202611172031,202611172032
    args.calibration_episodes,args.holdout_episodes=199,100
    args.batch_size,args.horizon,args.alpha=8,80,.05
    args.output_dir.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    actor=load_actor(args.package/'actor.pt').double().requires_grad_(False)
    old=ROOT/'final_cp/calibration.npz'
    dev=dict(np.load(old,allow_pickle=False))
    proto_old=json.loads((ROOT/'final_cp/protocol.json').read_text())
    assert proto_old['binding']['actor_sha256']==sha256(args.package/'actor.pt')
    assert {args.calibration_seed,args.holdout_seed}.isdisjoint({proto_old['calibration_seed'],proto_old['holdout_seed']})
    v,d=maxima(dev,actor)
    scale=dict(visual=np.quantile(v,.95,axis=0).tolist(),action=float(np.quantile(d,.95)))
    if min(*scale['visual'],scale['action'])<=0:
        raise ValueError('singular development scale')
    protocol=make_protocol(args)
    protocol['environment']['dtype']='float32 DINO and observer, float64 actor and plant'
    protocol.update(development=dict(file=str(old),sha256=sha256(old),episodes=len(d),scale=scale,
                                     rule='coordinate-wise empirical 95th percentiles of trajectory maxima'),
                    scoring=dict(uniform='max of both coordinates and time',diagonal='max coordinate / development scale',
                                 joint='max(0.5 * diagonal score, max exact clipped action difference / development action scale)'),
                    audit=dict(extra_normalized_input_error=MARGINS,grid=40,max_depth=16,
                               error_subdivisions=2,curvatures=core.KAPPAS.tolist() if isinstance(core.KAPPAS,np.ndarray) else list(core.KAPPAS),
                               max_pending=30000,method='affine intervals; equal numerical budget; clipped steering'),
                    joint_visual_weight=VISUAL_WEIGHT,comparison_source=sha256(Path(__file__)))
    protocol['score']='three score functions under scoring; one score per complete trajectory per arm'
    proto=args.output_dir/'protocol.json'
    write_json(proto,protocol)
    token=sha256(proto)
    camera=DevicePatchObserver(args.package/'patch_projection.pt',args.device,args.dino_repo,args.dino_weights)
    renderer=RoadRenderer(192,192)
    arms={}
    for split in ('calibration','holdout'):
        data=collect(args,split,DoubleAction(actor),camera,renderer,token)
        values=scores(maxima(data,actor),scale)
        if split=='calibration':
            for name,row in values.items():
                q,rank=conformal_radius(row,args.alpha)
                arms[name]=dict(q=q,rank=rank,**parameters(name,q,scale))
            cp=args.output_dir/'contracts.json'
            write_json(cp,dict(arms=arms,protocol=token,calibration=sha256(args.output_dir/'calibration.npz')))
            locked=sha256(cp)
            print('Frozen thresholds',arms,flush=True)
        else:
            metrics={name:dict(covered=int((row<=arms[name]['q']).sum()),n=len(row))
                     for name,row in values.items()}
            assert sha256(cp)==locked and sha256(proto)==token
            write_json(args.output_dir/'coverage.json',dict(arms=metrics,contracts_sha256=locked,
                       holdout_sha256=sha256(args.output_dir/'holdout.npz'),unsafe=int(data['unsafe'].sum()),
                       mean_abs_e_m=float(np.abs(data['states'][:,1:,0]).mean())))
            print(metrics,flush=True)


if __name__=='__main__':
    main()
