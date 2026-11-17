"""Evaluate curvature-conditioned observation proxies and retrained neural controllers."""
import argparse
from pathlib import Path
import multiprocessing as mp

import numpy as np
import torch

from . import evaluate as base
from ..lane_contract_comparison.collect import maxima,parameters,scores
from ..lane_keeping.cp import conformal_radius
from ..lane_keeping.learned_sbc import sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import DevicePatchObserver
from ..lane_keeping import core


STUDY='target_development_v2'
SEEDS=(202611178311,202611178312)
NETWORKS=('fixed_raw','fixed_corrected',*base.ARMS)


def offsets(ks, centers):
    return np.array([centers[str(float(k))] for k in ks])


class CorrectedActor(torch.nn.Module):
    def __init__(self, actor, centers):
        super().__init__()
        self.actor=actor
        self.register_buffer('keys',torch.tensor(base.KAPPAS,dtype=torch.float64)/core.KAPPA_SCALE)
        self.register_buffer('bias',torch.tensor(offsets(base.KAPPAS,centers),dtype=torch.float64))

    def forward(self, inputs):
        match=inputs[:,2:3]==self.keys
        if not (match.sum(1)==1).all():
            raise ValueError('unregistered curvature')
        shift=match.to(self.bias.dtype)@self.bias
        return self.actor(torch.cat((inputs[:,:2]-shift,inputs[:,2:]),1))


def paired_report(data):
    reference=data['uniform']['states'][:,0]
    curves=data['uniform']['kappas']
    values={}
    for name,z in data.items():
        np.testing.assert_array_equal(z['states'][:,0],reference)
        np.testing.assert_array_equal(z['kappas'],curves)
        values[name]=np.abs(z['states'][:,1:,0]).mean(axis=1)
    comparisons=[('fixed_raw','fixed_corrected'),('fixed_corrected','uniform'),
                 ('fixed_corrected','diagonal'),('fixed_corrected','joint'),
                 ('uniform','diagonal'),('uniform','joint'),('diagonal','joint')]
    out={}
    for a,b in comparisons:
        diff=values[b]-values[a]
        for group,selected in [('all',diff),('higher_curvature',diff[np.abs(curves)==.02])]:
            rng=np.random.default_rng(202611178315)
            means=rng.choice(selected,size=(4000,len(selected)),replace=True).mean(1)
            out[f'{b}_minus_{a}_{group}']=dict(delta_mean_abs_e_m=float(selected.mean()),
                   bootstrap95=np.quantile(means,[.025,.975]).tolist(),n=len(selected),
                   improved=int((selected<0).sum()))
    return out


def main(args):
    root=args.root
    train=base.read(root/'protocol.json')
    bias=base.read(root/'bias.json')
    centers=bias['centers']
    scale=bias['scale']
    if sha256(root/'bias.json')!=train['bias_sha256']:
        raise ValueError('different target-domain proxy')
    packages={name:base.PACKAGE if name.startswith('fixed_') else root/name for name in NETWORKS}
    bound={key:{n:sha256(folder/(n+'.pt')) for n in ['actor','sbc']} for key,folder in packages.items()}
    for arm in base.ARMS:
        frozen=base.read(root/arm/'frozen.json')
        if any(frozen[n+'_sha256']!=bound[arm][n] for n in ['actor','sbc']):
            raise ValueError('trained network changed')
    old=base.read(base.REFERENCE/'protocol.json')
    observer_path=base.PACKAGE/'patch_projection.pt'
    if sha256(observer_path)!=old['binding']['observer_sha256'] or sha256(base.WEIGHTS)!=old['binding']['dino_weights_sha256']:
        raise ValueError('visual assets differ from the frozen reference')
    study=dict(mode=STUDY,arms=NETWORKS,calibration_seed=SEEDS[0],holdout_seed=SEEDS[1],counts=base.SPLITS,
               kappas=base.KAPPAS,horizon=base.HORIZON,batch=args.batch,device=args.device,
               source=sha256(Path(__file__)),collector=sha256(Path(base.__file__)),models=bound,
               dino_weights=old['binding']['dino_weights_sha256'],observer=sha256(observer_path),
               baseline_protocol=sha256(base.REFERENCE/'protocol.json'),training_protocol=sha256(root/'protocol.json'),
               bias_sha256=sha256(root/'bias.json'),scale=scale,primary='mean abs lateral error at |kappa|=.02, all episodes and safety audits also reported',
               fixed_raw='old actor, original raw observation, no bias; all three original scoring scales from prior image protocol',
               fixed_corrected='same old actor with common curvature correction; all three target error contracts',
               trained='three retrained actors use identical correction; each gets own new closed-loop calibration and holdout',
               alternate_sign='negative paired difference = lower lateral error; percentile bootstrap whole trajectories conditional on systems')
    base.save(root/'image_protocol.json',study)
    token=sha256(root/'image_protocol.json')
    # The shared renderer/collector selects its independent draws via a study key.
    # Add a distinct key; the previously registered fixed and trained streams are preserved.
    base.SEEDS.setdefault(STUDY,SEEDS)
    assert base.SEEDS[STUDY]==SEEDS
    nets={key:load_actor(folder/'actor.pt').double().eval().requires_grad_(False) for key,folder in packages.items()}
    with mp.get_context('fork').Pool(args.workers) as pool:
        torch.set_num_threads(2)
        camera=DevicePatchObserver(observer_path,args.device,base.REPO,base.WEIGHTS).eval()
        fits,locks={},{}
        performance,held={},{}
        for split in base.SPLITS:
            for name, net in nets.items():
                raw=(name=='fixed_raw')
                interface=net if raw else CorrectedActor(net,centers).eval()
                folder=root/name
                folder.mkdir(parents=True,exist_ok=True)
                z=base.collect(root,STUDY,folder,split,interface,camera,pool,args.batch,token)
                centered=z if raw else dict(z,observations=z['observations']-offsets(z['kappas'],centers)[:,None,:],
                                             errors=z['errors']-offsets(z['kappas'],centers)[:,None,:])
                if not raw:
                    base.save_arrays(folder/(split+'_corrected.npz'),dict(corrected_observations=centered['observations'],
                                     centered_errors=centered['errors'],raw_sha256=np.array(sha256(folder/(split+'.npz')))))
                scales=old['development']['scale'] if raw else scale
                scored=scores(maxima(centered,net),scales)
                for arm in base.ARMS if name.startswith('fixed_') else (name,):
                    path=(folder/arm if name.startswith('fixed_') else folder)/'final_contract.json'
                    path.parent.mkdir(exist_ok=True)
                    key=f'{name}/{arm}' if name.startswith('fixed_') else name
                    if split=='calibration':
                        q,r=conformal_radius(scored[arm],.05)
                        fits[key]=dict(q=q,rank=r,**parameters(arm,q,scales),arm=arm,centered=(not raw),
                                       actor_sha256=bound[name]['actor'],sbc_sha256=bound[name]['sbc'],
                                       protocol=token,calibration=sha256(folder/'calibration.npz'))
                        base.save(path,fits[key])
                        locks[key]=sha256(path)
                    else:
                        if sha256(path)!=locks[key]: raise ValueError('threshold changed')
                        performance[key]=base.stats(z,scored[arm],fits[key]['q'])
                if split=='holdout':
                    held[name]=z
            if split=='calibration':
                base.save(root/'all_tubes_frozen.json',dict(contracts=locks,protocol=token))
                print('all target thresholds frozen',flush=True)
        base.save(root/'blind_summary.json',dict(arms=performance,paired=paired_report(held),contracts=locks,
                   image_protocol=token,holdout_sha256={key:sha256(root/key/'holdout.npz') for key in NETWORKS}))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('--batch',type=int,default=8)
    cli.add_argument('--workers',type=int,default=6)
    cli.add_argument('--device',default='cuda:1')
    main(cli.parse_args())
