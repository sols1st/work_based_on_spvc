"""Independently replay saved measurements and compare matched trajectories."""
import argparse
import hashlib
import math

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..curve_braking.vision import Encoder
from . import certificate, plant
from .compare_proxies import SOURCE, OUTPUT, REPO, WEIGHTS, read
from .compare_rollouts import parameters
from .perception import render


def bootstrap(delta):
    gen=np.random.default_rng(617219)
    means=delta[gen.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
    return dict(neural_minus_linear_mean=float(delta.mean()),
                interval_95=np.quantile(means,[.025,.975]).tolist(),
                lower=int((delta<0).sum()),higher=int((delta>0).sum()),n=len(delta))


@torch.inference_mode()
def run(source, output, replay_encoder=None):
    spec=read(output/'protocol.json')
    front=torch.load(source/'frontend.pt',weights_only=True,map_location='cpu')
    design=read(source/'evaluation/synthesis.json')['selected']
    assert design==spec['design']
    sealed=read(output/'all_tubes_frozen.json')
    models=read(output/'all_models_frozen.json')
    rows={}; distances={}; peak={}; motions={}; checked=0
    for arm,sha in models.items():
        folder=output/arm
        assert digest(folder/'model.pt')==sha
        assert digest(folder/'contract.json')==sealed[arm]
        contract=read(folder/'contract.json')
        model=parameters(folder,front)
        audit=certificate.verify(design,contract['q'])
        assert read(folder/'certificate.json')['exact_margins']==audit['exact_margins']
        assert read(folder/'certificate.json')['contract']==sealed[arm]
        for split in ('calibration','holdout'):
            n,seed=spec[split]
            x0, wind,light=plant.scenarios(n,seed)
            path=folder/f'{split}.npz'
            records=np.load(path,allow_pickle=False)
            stamp=read(folder/f'{split}_provenance.json')
            assert stamp['records']==digest(path)
            assert stamp['evaluation']==digest(output/'evaluation_protocol.json')
            if split=='holdout':
                assert stamp['contract']==digest(output/'all_tubes_frozen.json')
            else:
                assert contract['calibration']==digest(path)
            states=records['states']; y=records['latent']
            np.testing.assert_array_equal(states[:,0],x0)
            np.testing.assert_array_equal(records['lights'],light)
            np.testing.assert_array_equal(records['winds'],wind)
            positions=states[:,:-1,:3]; velocities=states[:,:-1,3:]
            x32=torch.tensor(positions,dtype=torch.float32).reshape(-1,3)
            y32=torch.from_numpy(y.reshape(-1,3))
            # Cross-check in double precision against the stored float32 inverse and nominal readings.
            m64=parameters(folder,front).double()
            decoded=m64.decode(y32.double()).reshape(y.shape).numpy()
            expected_raw=y - m64.nominal(x32.double()).reshape(y.shape).numpy()
            np.testing.assert_allclose(decoded-positions,records['errors'],rtol=0,atol=4e-7)
            np.testing.assert_allclose(expected_raw,records['raw'],rtol=0,atol=4e-7)
            action=plant.control(positions+records['errors'],velocities,design)
            np.testing.assert_allclose(action,records['actions'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(plant.step(states[:,:-1],action,wind),states[:,1:],rtol=0,atol=1e-12)
            score=np.max(np.abs(records['errors']),axis=(1,2))
            raw_score=np.max(np.abs(records['raw']),axis=(1,2))
            np.testing.assert_array_equal(records['scores'],score)
            for t in range(plant.HORIZON):
                image=render(positions[-1,t],light[-1])
                assert hashlib.sha256(image.tobytes()).hexdigest().encode()==records['images_sha256'][-1,t]
                if replay_encoder is not None:
                    b=spec['image_batch']; end=n//b*b
                    start=end if end<n else n-b
                    ids=np.minimum(np.arange(start,start+b),n-1)
                    images=[render(positions[i,t],light[i]) for i in ids]
                    f=replay_encoder(images)
                    z=(f-front['mean'])@front['basis']/front['std']
                    result=model.head(z)
                    np.testing.assert_allclose(result.numpy(),y[ids,t],rtol=0,atol=2e-5)
                    checked+=len(set(ids.tolist()))
            if split=='calibration':
                rank=math.ceil((len(score)+1)*.95)
                assert contract['q']==np.sort(score)[rank-1] and contract['rank']==rank
                assert contract['q_raw']==np.sort(raw_score)[rank-1]
            else:
                safe,hover,distance=plant.outcomes(states)
                cov=int((score<=contract['q']).sum())
                blind=read(folder/'blind.json')
                assert (int(safe.sum()),int(hover.sum()),cov)==(blind['safe'],blind['hover'],blind['covered'])
                distances[arm]=distance; peak[arm]=score
                motions[arm]=np.linalg.norm(states[:,:,:3]*plant.SCALE,axis=2).mean(axis=1)
                train=read(folder/'training.json')
                rows[arm]=dict(seed=train['seed'],kind=train['kind'],training=train['model'],chosen_epoch=train['chosen_epoch'],
                               trainable=train['trainable'],normalized_static_mse=train['test_decoded_mse'],
                               static_test_mae_m=train['test_mae_m'],geometry=train['geometry'],
                               q=contract['q'],q_raw=contract['q_raw'],covered=cov,safe=int(safe.sum()),hover=int(hover.sum()),
                               mean_final_m=float(distance.mean()),mean_tracking_distance_m=float(motions[arm].mean()),
                               physical_q_m=(contract['q']*plant.SCALE).tolist(),certificate_passed=audit['passed'],
                               transformed_velocity_margin=audit['margins']['transformed_velocity_update'])
    pair={}
    for seed in spec['seeds']:
        left,right=[f'{seed}_{k}' for k in ('linear','neural')]
        assert read(output/left/'training.json')['batches']==read(output/right/'training.json')['batches']
        pair[str(seed)]={name:bootstrap(metric[right]-metric[left]) for name,metric in
                         [('final_distance_m',distances),('trajectory_mean_distance_m',motions),('decoded_peak',peak)]}
    result=dict(arms=rows,paired=pair,calibration=spec['calibration'],holdout=spec['holdout'],
                signed_difference='neural minus linear, paired by entire scenario; bootstrap conditional on fitted models',
                verified_image_digests=len(models)*2*plant.HORIZON,
                oracle='exact rational coefficient inequalities for frozen PD and normalized decoded box')
    save(output/'summary.json',result)
    if replay_encoder is not None:
        save(output/'dino_recheck.json',dict(checked_measurements=checked,max_abs_tolerance=2e-5))
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    from pathlib import Path
    cli.add_argument('--source',type=Path,default=SOURCE)
    cli.add_argument('--output',type=Path,default=OUTPUT)
    cli.add_argument('--device',help='if supplied, also replay an image batch per control step and split with DINO')
    cli.add_argument('--repo',type=Path,default=REPO)
    cli.add_argument('--weights',type=Path,default=WEIGHTS)
    args=cli.parse_args()
    torch.set_num_threads(2)
    enc=Encoder(args.repo,args.weights,args.device) if args.device else None
    run(args.source,args.output,enc)
