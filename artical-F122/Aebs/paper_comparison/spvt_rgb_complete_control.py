"""Native SPVT controller checkpoints vs our actual RGB-DINO lane pipeline.

Original SPVT plant is used unchanged. Native published/trained weights remain
unchanged. RGB camera transfer and previous training histories are explicit.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from .proxy_architecture_compare import sha, write
from .spvt_native_checkpoint_study import load_native, metrics, summarize


BASE=Path('results/lane_precision_20261120_v1/202611200701/adversarial_precision/package')


def render_batch(states):
    from Aebs.lane_keeping.renderer import RoadRenderer
    renderer=RoadRenderer(192,192)
    return np.stack([renderer.render([x[0]/100,np.deg2rad(x[1])],0.,lighting=1.,
        occluder=False,camera_lateral=0.,camera_height=1.25,blur=0.,noise=0.) for x in states])


def native_pixels(images):
    return np.stack([np.asarray(Image.fromarray(im).convert('L').resize((16,8),Image.Resampling.BICUBIC),
        dtype=np.float32).reshape(-1)/127.5-1 for im in images])


def quantile(scores):
    rank=math.ceil((len(scores)+1)*.95)
    return float(np.sort(scores)[rank-1]) if rank<=len(scores) else float('inf')


def finite(value):
    return float(value) if np.isfinite(value) else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('/root/SPVT'))
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--calibration',type=int,default=199)
    p.add_argument('--test',type=int,default=1000)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    root=a.upstream.resolve();sys.path[:0]=[str(root),str(root/'spvt')]
    from model import model as module
    from spvt.utils import dynamics
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.robust_trajectory_cp import DevicePatchObserver
    selection=json.loads(a.selection.read_text())['selected']
    selected=Path(selection['path'])
    if sha(selected)!=selection['sha256']:raise ValueError('native selection binding')
    paths=dict(spvt_selected=selected,spvt_epoch100=root/'spvt/logs/test/ckpts/checkpoint_epoch_100.pth',
               pid_initializer=root/'model/pretrained_ckpts/pid_controller.pth')
    native={name:load_native(module,root,path,'cpu') for name,path in paths.items()}
    gen=native['pid_initializer'].generator
    ours=load_actor(BASE/'actor.pt').requires_grad_(False)
    repo=Path('../external/dinov2').resolve();weights=Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth').resolve()
    source=Path('Aebs/lane_keeping/renderer.py')
    inputs=[Path(__file__),a.selection,*paths.values(),root/'model/pretrained_ckpts/mlp_generator.pth',
            source,Path('Aebs/lane_keeping/patch_perception.py'),Path('Aebs/lane_keeping/policy.py'),
            root/'spvt/utils.py',root/'model/model.py',BASE/'actor.pt',BASE/'patch_projection.pt',BASE/'sbc.pt',weights]
    bindings={str(f):sha(f) for f in inputs}
    protocol=dict(version='native_spvt_to_rgb_lane_v1',smoke=a.smoke,
        native_implementation='unmodified native controllers, generator and dynamics; no optimization or source fixes',
        native_checkpoint_selection=dict(file_sha256=sha(a.selection),selected=selection),
        own='prespecified first training seed 202611200701 adversarial_precision; RGB DINO patch projection plus 3-64-64-1 neural controller',
        own_proxy='h(x)=[e/.85,heading_rad/.30]; known curvature=0',
        input='same RGB road rendering law at each method own current true state, new frame every time, not frozen-bank replay',
        native_preprocessing='RGB -> Pillow L -> BICUBIC 16x8 -> flatten128 -> [-1,1]; explicitly registered camera-transfer adapter because repository ships generated observation controller, not raw RGB loader',
        own_preprocessing='RGB192x192 -> native DINO RGB preprocessing -> patch tokens -> frozen spatial low-dimensional projection',
        physical_task='original SPVT dynamics, dt=.05s v=5m/s L=2.5m, steering +/-30degrees, eight seconds160steps',
        action_conversion='native u*30degrees; own normalized actor*.45rad then converted once to native u=phi/(pi/6)',
        initial='uniform [-60,60]cm x [-7,7]degrees',
        safety='unsafe iff any sampled |CTE|>80cm; no additional heading constraint',
        precision='safe whole episode and every sample from2 through8s within5cm or10cm',
        phases=dict(calibration=dict(n=a.calibration,seed=202611231101),test=dict(n=a.test,seed=202611231102)),
        horizon=160,batch=a.batch,workers=a.workers,device=a.device,
        freeze='all checkpoints fixed before any camera calibration; all calibration thresholds frozen before any test',
        visual_action_residual='actual camera action minus own nominal proxy action (G(0,x) for native, h(x) for ours); in degrees, not deviation from a new teacher',
        guarantees='do not transfer original method guarantees to this new rendering/plant condition; audit own barrier separately',
        resources='original SPVT pretrained/historical checkpoints versus our target-renderer trained front and controller; not equal-training-budget attribution',
        bindings=bindings,arms=list(native)+['current_rgb_dino'])
    write(a.output/'protocol.json',protocol)
    observer=DevicePatchObserver(BASE/'patch_projection.pt',a.device,repo,weights)
    all_results={};thresholds={};test_arrays={};start=time.time()
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        def render(states):
            chunks=[part for part in np.array_split(states,min(len(states),a.workers)) if len(part)]
            return np.concatenate(list(pool.map(render_batch,chunks)))

        # Renderer dispatch is checked against the reference RGB renderer.
        probe=np.array([[-60,-7],[0,0],[60,7],[25,-2]],dtype=np.float32)
        expected=render_batch(probe);actual=render(probe)
        if not np.array_equal(expected,actual):raise ValueError('parallel pixel mismatch')
        write(a.output/'pixel_equivalence.json',dict(equal=True,cases=4,
            raw_sha256=hashlib.sha256(actual.tobytes()).hexdigest()))
        for phase,setup in protocol['phases'].items():
            rng=np.random.default_rng(setup['seed']);initial=rng.uniform([-60,-7],[60,7],(setup['n'],2)).astype('float32')
            for name in protocol['arms']:
                parts=[];armstart=time.time()
                for first in range(0,len(initial),a.batch):
                    x=torch.tensor(initial[first:first+a.batch]);count=len(x)
                    states=[x.numpy().copy()];actions=[];gaps=[];ob_errors=[];hashes=[hashlib.sha256() for _ in range(count)]
                    for t in range(160):
                        images=render(x.numpy())
                        for i in range(count):hashes[i].update(images[i].tobytes())
                        with torch.inference_mode():
                            if name=='current_rgb_dino':
                                observed=observer(torch.from_numpy(images))
                                nominal=torch.stack([x[:,0]/85,torch.deg2rad(x[:,1])/.30],1)
                                zero=torch.zeros(count,1)
                                u=ours(torch.cat([observed,zero],1)).flatten()*.45/(math.pi/6)
                                nominal_u=ours(torch.cat([nominal,zero],1)).flatten()*.45/(math.pi/6)
                                ob_errors.append((observed-nominal).numpy())
                            else:
                                m=native[name]
                                u=m.controller(torch.from_numpy(native_pixels(images))).flatten()
                                nominal_u=m(torch.zeros(count,10),x).flatten()
                            if not torch.isfinite(u).all():raise ValueError('nonfinite controller output')
                            actions.append(u.numpy());gaps.append(((u-nominal_u)*30).numpy())
                            e,h=dynamics(x[:,0],x[:,1],u)
                            x=torch.stack([e,h],1)
                        states.append(x.numpy().copy())
                        if first==0 and t in [0,40,159]:
                            directory=a.output/'examples';directory.mkdir(exist_ok=True)
                            Image.fromarray(images[0]).save(directory/f'{name}__{phase}__t{t:03d}.png')
                            Image.fromarray(images[0]).convert('L').resize((16,8),Image.Resampling.BICUBIC).save(directory/f'{name}__{phase}__t{t:03d}__native_input.png')
                    state=np.stack(states,1);action=np.stack(actions,1);gap=np.stack(gaps,1)
                    metric=metrics(state,action)
                    metric['visual_action_score_deg']=abs(gap).max(1)
                    # Joint statistical safety/precision score; unsafe -> infinity.
                    metric['performance_score_cm']=np.where(metric['unsafe'],np.inf,abs(state[:,40:,0]).max(1))
                    part=dict(states_cm_degrees=state,actions_normalized=action,visual_action_gap_degrees=gap,
                              image_digest=np.array([h.hexdigest() for h in hashes]),**metric)
                    if ob_errors:
                        error=np.stack(ob_errors,1);part.update(observation_errors=error,observation_score=abs(error).max(axis=(1,2)))
                    parts.append(part)
                    print(json.dumps(dict(phase=phase,arm=name,completed=first+count,n=len(initial),seconds=time.time()-armstart)),flush=True)
                combined={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                np.savez_compressed(a.output/(name+'__'+phase+'.npz'),**combined)
                if phase=='calibration':
                    thresholds[name]=dict(action_q_deg=quantile(combined['visual_action_score_deg']),
                        performance_q_cm=finite(quantile(combined['performance_score_cm'])),
                        unsafe_calibration=int(combined['unsafe'].sum()))
                    if name=='current_rgb_dino':thresholds[name]['observation_q']=quantile(combined['observation_score'])
                else:
                    row=summarize({k:combined[k] for k in metrics(combined['states_cm_degrees'],combined['actions_normalized'])})
                    q=thresholds[name]
                    row.update(n=len(initial),unsafe_count=int(combined['unsafe'].sum()),
                        strict_5cm_count=int(combined['strict_5cm'].sum()),strict_10cm_count=int(combined['strict_10cm'].sum()),
                        thresholds=q,action_covered=int((combined['visual_action_score_deg']<=q['action_q_deg']).sum()),
                        performance_covered=None if q['performance_q_cm'] is None else int((combined['performance_score_cm']<=q['performance_q_cm']).sum()))
                    if name=='current_rgb_dino':row['observation_covered']=int((combined['observation_score']<=q['observation_q']).sum())
                    all_results[name]=row;test_arrays[name]=combined
                    print(json.dumps(dict(phase='test_summary',arm=name,metrics=row)),flush=True)
            if phase=='calibration':write(a.output/'frozen_thresholds.json',thresholds)
    # Pair whole scenarios; do not treat time samples as independent tests.
    paired={};rng=np.random.default_rng(202611231103)
    ours=test_arrays['current_rgb_dino']
    for name in native:
        old=test_arrays[name];np.testing.assert_array_equal(old['states_cm_degrees'][:,0],ours['states_cm_degrees'][:,0])
        differences={k:ours[k].astype(float)-old[k].astype(float) for k in ['rmse_cm','strict_5cm','unsafe']}
        paired[name]={k:dict(mean=float(v.mean()),ci95=np.quantile(rng.choice(v,(4000,len(v))).mean(1),[.025,.975]).tolist()) for k,v in differences.items()}
    assert all(sha(f)==bindings[str(f)] for f in inputs)
    write(a.output/'summary.json',dict(arms=all_results,paired_ours_minus_native=paired,elapsed_seconds=time.time()-start))


if __name__=='__main__':main()
