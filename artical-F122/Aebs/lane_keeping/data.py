"""Generate the lane-keeping image dataset with scene-group splits.

Each scene fixes curvature, lighting, camera pose and optional occlusion, and
samples states on a coarse grid plus random points.  Scenes -- not frames --
are split into train/validation/calibration/test, so images from one condition
never leak across splits.  This matters for both the DINO projection and the
split-conformal calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from . import core
from .renderer import RoadRenderer

SPLIT_FRACTIONS = (('train', .55), ('validation', .15), ('calibration', .20), ('test', .10))
KAPPA_CHOICES = (-.01, 0., .01)
LIGHTING = (.9, 1., 1.1)
CAMERA_LATERAL = (-.1, 0., .1)
CAMERA_HEIGHT = (1.2, 1.3)


def make_scenes(n_scenes, seed=0, randomized=False):
    """Nominal scenes by default; randomized appearance is an OOD stress mode.

    Nominal mode keeps lighting/camera/occluder fixed and varies only curvature
    and the sampled states.  This is the distribution the CP contract covers.
    ``randomized=True`` is for separate distribution-shift stress tests.
    """
    if n_scenes < 3:
        raise ValueError('at least three scenes required')
    rng = np.random.default_rng(seed)
    scenes = []
    if not randomized:
        per_kappa = max(1, n_scenes // len(KAPPA_CHOICES))
        index = 0
        for kappa in KAPPA_CHOICES:
            for j in range(per_kappa):
                scenes.append(dict(scene_id=index, kappa=float(kappa), lighting=1.,
                                   occluder=False, camera_lateral=0., camera_height=1.25,
                                   blur=0., noise=0., appearance='nominal'))
                index += 1
        # Deterministic per-kappa split: 5 train, 1 validation, 1 calibration, 1 test
        # when there are eight scenes per curvature.  Adjust proportionally.
        for kappa in KAPPA_CHOICES:
            group = [s for s in scenes if s['kappa'] == float(kappa)]
            n = len(group)
            cuts = [max(1, int(round(.625 * n))), max(2, int(round(.75 * n))),
                    max(3, int(round(.875 * n)))]
            for i, scene in enumerate(group):
                scene['split'] = ('train' if i < cuts[0] else
                                  'validation' if i < cuts[1] else
                                  'calibration' if i < cuts[2] else 'test')
        return scenes
    for index in range(n_scenes):
        scenes.append(dict(
            scene_id=index,
            kappa=float(KAPPA_CHOICES[index % len(KAPPA_CHOICES)]),
            lighting=float(rng.choice(LIGHTING)),
            occluder=bool(rng.random() < .25),
            camera_lateral=float(rng.choice(CAMERA_LATERAL)),
            camera_height=float(rng.choice(CAMERA_HEIGHT)),
            blur=float(rng.choice((0., .5))),
            noise=float(rng.choice((0., 2.))),
            appearance='randomized'))
    order = rng.permutation(n_scenes)
    split_of = {}
    start = 0
    for name, fraction in SPLIT_FRACTIONS:
        count = max(1, int(round(fraction * n_scenes)))
        for scene_id in order[start:start + count]:
            split_of[int(scene_id)] = name
        start += count
    for scene_id in order[start:]:
        split_of[int(scene_id)] = 'calibration'
    for scene in scenes:
        scene['split'] = split_of[scene['scene_id']]
    return scenes


def scene_states(scene, per_scene, seed=0):
    """Mostly random states: a coarse grid alone lets the projection overfit
    grid points and fail badly between them, which is exactly what the SBC
    verifier will query.  Keep a few structured points for coverage."""
    rng = np.random.default_rng(seed + 1000 * scene['scene_id'])
    structured = np.array([[-.8, -.25], [-.8, .25], [.8, -.25], [.8, .25],
                           [0., 0.], [0., -.25], [0., .25], [-.8, 0.], [.8, 0.]])
    n_random = max(0, per_scene - len(structured))
    random = rng.uniform([-.8, -.25], [.8, .25], size=(n_random, 2))
    return np.vstack([structured, random])


def generate(output_dir, n_scenes=24, per_scene=40, image_size=192, seed=0,
             randomized=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    scenes = make_scenes(n_scenes, seed, randomized)
    renderer = RoadRenderer(image_size, image_size)
    images, states, kappas, scene_ids = [], [], [], []
    for scene in scenes:
        points = scene_states(scene, per_scene, seed)
        for point in points:
            images.append(renderer.render(point, scene['kappa'], scene['lighting'],
                                          scene['occluder'], scene['camera_lateral'],
                                          scene['camera_height'], scene['blur'],
                                          scene['noise'], seed=scene['scene_id']))
            states.append(point)
            kappas.append(scene['kappa'])
            scene_ids.append(scene['scene_id'])
    images = np.asarray(images, dtype=np.uint8)
    states = np.asarray(states, dtype=np.float32)
    kappas = np.asarray(kappas, dtype=np.float32)
    scene_ids = np.asarray(scene_ids, dtype=np.int32)
    split_of = {scene['scene_id']: scene['split'] for scene in scenes}
    split_names = np.asarray([split_of[int(i)] for i in scene_ids])
    arrays = dict(images=images, states=states, kappas=kappas, scene_ids=scene_ids,
                  split=split_names)
    np.savez(output_dir / 'dataset.npz', **arrays)
    (output_dir / 'scenes.json').write_text(json.dumps(scenes, indent=2))
    (output_dir / 'metadata.json').write_text(json.dumps(dict(
        n_scenes=n_scenes, per_scene=per_scene, image_size=image_size, seed=seed,
        randomized=randomized,
        scales=dict(e=core.E_SCALE, theta=core.THETA_SCALE, kappa=core.KAPPA_SCALE),
        counts={name: int(np.sum(split_names == name)) for name in
                ('train', 'validation', 'calibration', 'test')}), indent=2))
    print(json.dumps(dict(images=images.shape, states=states.shape,
                          counts={name: int(np.sum(split_names == name))
                                  for name in ('train', 'validation', 'calibration', 'test')}),
                     indent=2))
    return output_dir


def load_dataset(path):
    data = np.load(Path(path) / 'dataset.npz')
    return {key: data[key] for key in data.files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scenes', type=int, default=24)
    parser.add_argument('--states-per-scene', type=int, default=40)
    parser.add_argument('--image-size', type=int, default=192)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--randomized', action='store_true',
                        help='distribution-shift stress scenes, not the CP training distribution')
    args = parser.parse_args(argv)
    generate(args.output_dir, args.scenes, args.states_per_scene, args.image_size,
             args.seed, args.randomized)


if __name__ == '__main__':
    main()
