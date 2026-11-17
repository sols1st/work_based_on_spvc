"""Independently replay stored crane transitions, verify calibrated radii, and compare paired outcomes."""
import argparse
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..curve_braking.evaluate import digest, save
from ..experiments.analyze_crane_fixed_contracts import exact_geometry
from ..overhead_crane import certificate, plant
from ..overhead_crane.camera import render
from .train import FAMILIES, SEEDS, ProxyObservation, arm_path, read


def per_scene(states):
    load = states[...,0] + states[...,2]
    obs = dict(trolley_position=states[...,0], trolley_speed=states[...,1], sway=states[...,2],
               sway_rate=states[...,3], payload_position=load, payload_diagonal=load.sum(-1,keepdims=True))
    safe = np.logical_and.reduce([np.abs(obs[k]).max((1,2)) < v for k,v in plant.LIMITS.items()])
    dwell = np.logical_and.reduce([np.abs(obs[k][:,-plant.DWELL:]).max((1,2)) <= v for k,v in plant.TARGET.items()])
    return dict(safe=safe, settled=safe&dwell, final_load_m=np.linalg.norm(load[:,-1],axis=-1),
                last_second_sway_m=np.linalg.norm(states[:,-7:,:,2],axis=-1).mean(-1))


@torch.inference_mode()
def replay(data, model, n, seed, gains, verify_image):
    states = data['states']
    starts, noise, lights = plant.scenarios(n, seed)
    for key, expected in [('noise',noise),('lights',lights)]:
        np.testing.assert_array_equal(data[key], expected)
    np.testing.assert_array_equal(starts, states[:,0])
    z = plant.observation(states[:,:-1])
    t = lambda a: torch.tensor(a.reshape(-1,4),dtype=torch.float32)
    zh = model.decode(t(data['latent'])).numpy().reshape(z.shape)
    raw = data['latent'] - model.nominal(t(z)).numpy().reshape(z.shape)
    # CPU/native CUDA matrix solves can differ at single-precision roundoff.
    np.testing.assert_allclose(zh-z, data['errors'], rtol=0, atol=1.2e-6)
    np.testing.assert_allclose(raw, data['latent_errors'], rtol=0, atol=1.2e-6)
    actions = plant.control(z+data['errors'],states[:,:-1],gains)
    np.testing.assert_allclose(actions,data['actions'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(plant.step(states[:,:-1],actions,noise),states[:,1:],rtol=0,atol=1e-12)
    scores = np.abs(data['errors']).max((1,2))
    np.testing.assert_array_equal(data['scores'],scores)
    np.testing.assert_array_equal(data['latent_scores'],np.abs(data['latent_errors']).max((1,2)))
    if verify_image:
        i=n-1
        h = hashlib.sha256()
        for zt in z[i]:
            h.update(render(zt,lights[i]).tobytes())
        assert data['image_digests'][i] == h.hexdigest()
    return scores


def contrast(values, ix):
    d=np.asarray(values)
    if d.ndim != 2 or d.shape != (len(SEEDS),100):
        raise ValueError('Expected three paired models measured on 100 shared scenes')
    scene=d.mean(0)
    lower,upper=np.quantile(scene[ix].mean(1),[.025,.975])
    return dict(mean_delta=float(scene.mean()), interval95=[float(lower),float(upper)],
                scene_pairs_lower=int((scene<0).sum()),scene_pairs_higher=int((scene>0).sum()),
                by_training_seed=d.mean(1).tolist())


def check(root, verify_image):
    p=read(root/'protocol.json')
    seal=read(root/'all_contracts.json')
    assert seal['protocol']==digest(root/'protocol.json')
    rows={}
    values={}
    for seed in SEEDS:
        for family in FAMILIES:
            arm=arm_path(root,family,seed)
            fit,summary,trained=[read(arm/f) for f in ('certificate.json','summary.json','training.json')]
            assert digest(arm/'certificate.json')==seal['contracts'][arm.name]==summary['certificate']
            assert digest(arm/'frontend.pt')==trained['checkpoint']==p['models'][arm.name]['checkpoint']==fit['checkpoint']
            assert digest(arm/'training.json')==p['models'][arm.name]['training']
            assert digest(arm/'calibration.npz')==fit['calibration']
            assert digest(arm/'holdout.npz')==summary['test']
            model=ProxyObservation(family)
            model.load_state_dict(torch.load(arm/'frontend.pt',map_location='cpu',weights_only=True))
            model.eval()
            with np.load(arm/'calibration.npz') as data:
                scores=replay(data,model,*p['splits']['calibration'],p['gains'],verify_image)
                rank=int(np.ceil((len(scores)+1)*(1-p['alpha'])))
                q=float(np.sort(scores)[rank-1])
                assert fit['q']==q and rank==fit['rank']==190
                assert float(np.sort(data['latent_scores'])[rank-1])==fit['raw_q']
            audit=certificate.verify(p['gains'],q)
            assert audit==fit['audit']
            geometry=exact_geometry(p['gains'],q,Fraction(1))
            assert fit['geometry']==geometry
            with np.load(arm/'holdout.npz') as data:
                scores=replay(data,model,*p['splits']['holdout'],p['gains'],verify_image)
                sc=per_scene(data['states'])
                sc.update(peak_decoded=scores, covered=scores<=q)
                values[(family,seed)]={k:np.array(v,dtype=float) for k,v in sc.items()}
                assert int(sc['safe'].sum())==summary['safe'] and int(sc['settled'].sum())==summary['settled']
                assert int(sc['covered'].sum())==summary['covered']
                assert int((data['latent_scores']<=fit['raw_q']).sum())==summary['raw_covered']
                assert abs(np.mean(sc['final_load_m'])-summary['mean_final_load_m'])<1e-12
            rows[arm.name]=dict(family=family,seed=seed,epoch=trained['best_epoch'],
                normal_static_rmse=trained['normalized_decoded_test_rmse'], affine=trained['best_affine_test_rmse'],
                inverse=trained['inverse_roundtrip_max'], raw_q=fit['raw_q'],q=q,covered=summary['covered'],
                passed=audit['passed'],upper=audit['initial_upper'],scale=geometry['scale_supremum'],
                settled=summary['settled'],safe=summary['safe'],final_load_m=summary['mean_final_load_m'],
                sway_m=summary['mean_last_second_sway_m'])
    contrasts={}
    ix=np.random.default_rng(817751).integers(0,100,(10000,100))
    for key in next(iter(values.values())):
        ds=[values['neural',s][key]-values['linear',s][key] for s in SEEDS]
        contrasts[key]=contrast(ds,ix)
    result=dict(rows=rows,contrasts_neural_minus_linear=contrasts,training_seeds=SEEDS,
                physical_scales_m=[1.,1.,.5,.5], trajectory_counts={'calibration':1194,'blind':600},
                effective_rendered_frames=71760,padded_rendered_frames=6*(200+104)*plant.HORIZON,
                uncertainty='Paired bootstrap of the 100 shared blind scenarios, conditional on the three trained pairs.',
                checked=dict(complete_trajectories=1794,plant_transitions=71760,
                image_hashes_reconstructed=(12 if verify_image else 0)))
    return result


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    torch.set_num_threads(2)
    result=check(args.root,True)
    if args.output:
        save(args.output,result)
    print(result['checked'])
    for name,row in result['rows'].items():
        print(name,row['q'],row['covered'],row['upper'],row['settled'],row['final_load_m'])
