"""Replay the saved trajectories and independently check ranks, containment and wind inequalities."""
import argparse
import hashlib

import numpy as np
import torch

from ..conformal.trajectory_boxes import cuts, widths
from ..curve_braking.evaluate import digest, save
from ..curve_braking.vision import Encoder
from . import plant
from .compare_rollouts import parameters
from .collect_box_trajectories import OUT, SPLITS
from .compare_proxies import OUTPUT as PREVIOUS, REPO, SOURCE, WEIGHTS, read
from .cp_boxes import DESIGN, audit, features
from .perception import render


def containment(data, shape):
    pos = data['states'][:, :-1, :3]
    feat = features(data['states'][:, :-1])
    low, high = widths(shape,feat)
    err = data['errors']
    score = np.max(np.where(err < 0, -err / low, err / high),axis=(1,2))
    assert score.shape == (len(pos),)
    return score


def stress_shapes(shape, q, wind):
    # Separate numerical cross-check at both sides of each fitted knot, combined
    # with a regular mesh. The continuous proof is the rational cell audit.
    l,m,S = [DESIGN[k] for k in ['l','m','s_limit']]
    zcuts,tcuts = cuts(shape)
    def grid(values, lo, hi):
        edges=np.array(values,dtype=float)
        return np.unique(np.clip(np.r_[np.linspace(lo,hi,49), edges,
                                     np.nextafter(edges,-np.inf),np.nextafter(edges,np.inf)],lo,hi))
    z,t = np.meshgrid(grid(zcuts,-plant.POS_LIMIT,plant.POS_LIMIT),
                      grid([S,*tcuts],0,plant.POS_LIMIT*S))
    f = np.stack([z,t],axis=-1)
    neg,pos=widths(shape,f)
    lo,hi=q*neg,q*pos
    r=np.maximum(1,np.maximum(np.abs(z),t/S))
    budget=r[...,None]*S-abs(1-plant.DT*m)*t[...,None]
    needed=plant.DT*(wind + l*m*np.maximum(lo,hi))
    if np.min(budget-needed)<-4e-13:
        raise ValueError('Numerical endpoint check exceeds transformed-speed envelope')
    pmin=np.stack([np.full(z.shape,-plant.POS_LIMIT),np.full(z.shape,-plant.POS_LIMIT),z],-1)
    pmax=np.stack([np.full(z.shape,plant.POS_LIMIT),np.full(z.shape,plant.POS_LIMIT),z],-1)
    accel=(l+m)*t[...,None]+np.maximum(l*l*pmax+l*m*lo,-l*l*pmin+l*m*hi)
    if np.max(accel)>plant.ACCEL+4e-13:
        raise ValueError('Numerical endpoint check activates saturation')
    return z.size


@torch.inference_mode()
def check(replay):
    root = OUT
    seal = read(root/'all_boxes_frozen.json')
    comp = read(root/'comparison.json')
    spec = read(root/'collection_protocol.json')
    assert spec['design']==DESIGN
    assert digest(root/'all_boxes_frozen.json') == comp['contracts_sha256']
    front = torch.load(SOURCE/'frontend.pt',map_location='cpu',weights_only=True)
    counts={'observations':0,'image_digests':0,'dino_replays':0,'ranks_and_holdouts':0,'paired_trajectories':0,'wind_grid_state_pairs':0}
    for arm, key in spec['weights'].items():
        old = PREVIOUS/arm
        folder = root/arm
        model = parameters(old,front)
        assert digest(old/'model.pt')==key
        boxes=read(folder/'boxes.json')
        assert digest(folder/'boxes.json')==seal['contracts'][arm]
        assert boxes['development_sha256']==digest(old/'calibration.npz')
        runs={}
        for split,(n,seed) in SPLITS.items():
            file=folder/(split+'.npz')
            runs[split]=data=dict(np.load(file,allow_pickle=False))
            start,wind,light=plant.scenarios(n,seed)
            stamp=read(folder/(split+'_source.json'))
            assert stamp['sha256']==digest(file) and stamp['protocol']==digest(root/'collection_protocol.json')
            assert (stamp['seed'],stamp['n'],stamp['model'])==(seed,n,key)
            assert stamp['seal']==(digest(root/'all_boxes_frozen.json') if split=='holdout' else None)
            np.testing.assert_array_equal(data['states'][:,0],start)
            np.testing.assert_array_equal(data['winds'],wind)
            np.testing.assert_array_equal(data['lights'],light)
            pos=data['states'][:, :-1, :3]
            y=torch.from_numpy(data['latent'])
            z=model.decode(y.reshape(-1,3)).numpy().reshape(pos.shape)
            np.testing.assert_allclose(z-pos,data['errors'],rtol=0,atol=4e-7)
            acts=plant.control(pos+data['errors'],data['states'][:, :-1, 3:],DESIGN)
            np.testing.assert_allclose(acts,data['actions'],rtol=0,atol=2e-12)
            np.testing.assert_allclose(plant.step(data['states'][:, :-1],acts,wind),data['states'][:,1:],
                                       rtol=0,atol=2e-12)
            counts['observations']+=n*plant.HORIZON
            counts['paired_trajectories']+=n
            # Reconstruct the last real inference batch including its original padded rows.
            b = spec['batch']
            start = ((n-1)//b)*b
            ids = np.minimum(np.arange(start,start+b),n-1)
            for k in range(plant.HORIZON):
                imgs=[render(pos[i,k],light[i]) for i in ids]
                for i,img in zip(ids[:n-start],imgs):
                    assert hashlib.sha256(img.tobytes()).hexdigest().encode() == data['images_sha256'][i,k]
                counts['image_digests']+=n-start
                if replay is not None:
                    f = replay(imgs)
                    coords = ((f-front['mean'])@front['basis']) / front['std']
                    np.testing.assert_allclose(model.head(coords).numpy(),data['latent'][ids,k],rtol=0,atol=2e-5)
                    counts['dino_replays']+=n-start
        for key, spec_box in boxes['boxes'].items():
            shape=spec_box['shape']; q=spec_box['q']
            assert np.sort(containment(runs['calibration'],shape))[189]==q
            expected = comp['arms'][arm]['boxes'][key]
            assert int((containment(runs['holdout'],shape)<=q).sum())==expected['covered']
            recomputed=audit(shape,q)
            assert recomputed['exact_wind_bound']==expected['audit']['exact_wind_bound']
            assert recomputed['passed']==expected['audit']['passed']
            counts['wind_grid_state_pairs']+=stress_shapes(shape,q,recomputed['certification_metric'])
            counts['ranks_and_holdouts']+=1
    return counts


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device')
    args=cli.parse_args();torch.set_num_threads(2)
    encoder=Encoder(REPO,WEIGHTS,args.device) if args.device else None
    result=check(encoder)
    save(OUT/('dino_recheck.json' if args.device else 'independent_check.json'),result)
    print(result)
