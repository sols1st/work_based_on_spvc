"""Recreate fixed holdout scene0 only in the original renderer environment.

Raw RGB image hashes across all80 camera instants MUST match the recorded
rollout digest before any selected frame is exported for a paper figure.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import numpy as np
from PIL import Image
from Aebs.lane_keeping.renderer import RoadRenderer


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;steps=[0,20,79];renderer=RoadRenderer(192,192)
    report=dict(scene_index=0,selected_steps=steps,selection='fixed before inspecting performance',
        versions={name:metadata.version(name) for name in ['numpy','Pillow']},
        renderer_sha256=sha(Path(__file__).parents[1]/'lane_keeping/renderer.py'),
        reconstruction_source_sha256=sha(__file__),arms={},
        scope='Raw RGB bytes reconstructed in the collection environment, checked against the original80-frame digest; PNG bytes have their own hashes.')
    exported={}
    for arm in ['anchor','spvt_first_seed']:
        path=root/arm/'test.npz';data=np.load(path,allow_pickle=False)
        digest=hashlib.sha256();frames=[]
        for t in range(80):
            pixels=renderer.render(data['states'][0,t],float(data['kappas'][0]),lighting=1.,occluder=False,
                 camera_lateral=0.,camera_height=1.25,blur=0.,noise=0.)
            digest.update(pixels.tobytes())
            if t in steps:frames.append(pixels.copy())
        expected=str(data['images_sha256'][0]);computed=digest.hexdigest()
        report['arms'][arm]=dict(test_sha256=sha(path),recorded80_frame_sha256=expected,
            reconstructed80_frame_sha256=computed,full_trajectory_hash_matches=expected==computed,
            curvature=float(data['kappas'][0]),initial_state=data['states'][0,0].tolist())
        exported[arm]=np.stack(frames)
    report['all_match']=all(row['full_trajectory_hash_matches'] for row in report['arms'].values())
    if not report['all_match']:
        (root/'scene0_reconstruction.json').write_text(json.dumps(report,indent=2))
        raise AssertionError('Original recorded camera digest not reproduced; do not label selected images exact')
    np.savez_compressed(root/'scene0_images.npz',steps=np.array(steps),**exported)
    destination=root/'scene0_original_png';destination.mkdir(exist_ok=True)
    for arm,frames in exported.items():
        report['arms'][arm]['frames']=[]
        for step,pixels in zip(steps,frames):
            output=destination/f'{arm}_step{step:02d}.png';Image.fromarray(pixels).save(output)
            report['arms'][arm]['frames'].append(dict(step=step,png=str(output.relative_to(root)),png_sha256=sha(output),
                raw_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    report['image_archive_sha256']=sha(root/'scene0_images.npz')
    (root/'scene0_reconstruction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
