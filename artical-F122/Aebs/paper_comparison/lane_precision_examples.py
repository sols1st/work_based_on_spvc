"""Reconstruct registered scene zero's actual camera frames and verify hashes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

from .lane_precision import ARMS, SEEDS, save


def main():
    from Aebs.lane_keeping.renderer import RoadRenderer
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=Path, required=True)
    a = p.parse_args()
    out = a.camera/'camera_examples'
    out.mkdir(exist_ok=False)
    records = []
    for arm in ['anchor']+[str(SEEDS[0])+'__'+name for name in ARMS]:
        data = np.load(a.camera/arm/'holdout.npz')
        renderer, digest = RoadRenderer(192, 192), hashlib.sha256()
        images = []
        for t in range(80):
            image = renderer.render(data['states'][0, t], float(data['kappas'][0]), lighting=1.,
                occluder=False, camera_lateral=0., camera_height=1.25, blur=0., noise=0.)
            digest.update(image.tobytes())
            images.append(image)
        if digest.hexdigest() != str(data['images_sha256'][0]):
            raise AssertionError('reconstructed pixels differ from actual camera inputs')
        for t in [0, 20, 79]:
            filename = arm+'__step%02d.png'%t
            Image.fromarray(images[t]).save(out/filename)
            records.append(dict(arm=arm, scene=0, step=t, seconds=.1*t, filename=filename,
                physical_state=data['states'][0, t].tolist(), observation=data['observations'][0, t].tolist(),
                entire_episode_image_sha256=digest.hexdigest(), all_80_frames_hash_verified=True))
    save(out/'provenance.json', dict(selection='first prespecified test scene and first training seed, independent of outcomes',
                                  frames=records))


if __name__ == '__main__':
    main()
