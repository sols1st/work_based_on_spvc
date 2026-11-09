"""Closed-loop evaluation with real rendered images and the frozen DINO observer.

Three arms:
  oracle          semantic actor on the true normalized state
  image           image-trained actor on DINO observations
  image_oracle    semantic actor on DINO observations (perception gap only)
Stress scenes use randomized lighting/camera/occlusion and are reported
separately from the nominal CP-covered distribution.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .data import make_scenes, scene_states
from .env import summarise
from .perception import ImageObserver
from .patch_perception import PatchImageObserver
from .policy import load_actor
from .residual import load_residual
from .renderer import RoadRenderer


def _observation(actor, observation):
    with torch.no_grad():
        return actor(torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)).numpy().reshape(-1)


def image_rollout(actor, observer, renderer, scene, cp_radius, episodes=10, seed=0):
    rng = np.random.default_rng(seed + 1000 * scene['scene_id'])
    records = []
    for episode in range(episodes):
        state = core.sample_initial(rng, 1)[0]
        kappa = scene['kappa']
        unsafe_seen = False
        states, residuals, actions = [], [], []
        for step in range(core.HORIZON):
            image = renderer.render(state, kappa, scene['lighting'], scene['occluder'],
                                    scene['camera_lateral'], scene['camera_height'],
                                    scene['blur'], scene['noise'], seed=scene['scene_id'] * 100 + step)
            image_tensor = torch.from_numpy(image).unsqueeze(0)
            with torch.no_grad():
                observation = observer(image_tensor).numpy().reshape(-1)
            truth = core.state_to_net(state)
            residuals.append(np.abs(observation - truth).max())
            action = _observation(actor, np.concatenate([observation, [kappa / core.KAPPA_SCALE]]))[0]
            state = core.dynamics(state, core.action_from_net(action), kappa)
            states.append(state.copy())
            actions.append(float(action))
            unsafe_seen = unsafe_seen or core.unsafe(state)
            if unsafe_seen:
                break
        states = np.asarray(states)
        records.append(dict(episode=episode, success=bool(not unsafe_seen and len(states) >= core.HORIZON),
                            unsafe=bool(unsafe_seen),
                            margin=float(min(core.margin(s) for s in states)),
                            mean_abs_e=float(np.mean(np.abs(states[:, 0]))),
                            max_abs_e=float(np.max(np.abs(states[:, 0]))),
                            mean_residual=float(np.mean(residuals)),
                            cp_coverage=float(np.mean(np.asarray(residuals) <= cp_radius)),
                            kappa=float(kappa), states=states.tolist()))
    return records


def semantic_rollout(actor, scene, episodes=10, seed=0):
    rng = np.random.default_rng(seed + 2000 * scene['scene_id'])
    records = []
    for episode in range(episodes):
        state = core.sample_initial(rng, 1)[0]
        kappa = scene['kappa']
        unsafe_seen = False
        states = []
        for step in range(core.HORIZON):
            observation = core.to_net(state, kappa)
            action = _observation(actor, observation)[0]
            state = core.dynamics(state, core.action_from_net(action), kappa)
            states.append(state.copy())
            unsafe_seen = unsafe_seen or core.unsafe(state)
            if unsafe_seen:
                break
        states = np.asarray(states)
        records.append(dict(episode=episode, success=bool(not unsafe_seen and len(states) >= core.HORIZON),
                            unsafe=bool(unsafe_seen),
                            margin=float(min(core.margin(s) for s in states)),
                            mean_abs_e=float(np.mean(np.abs(states[:, 0]))),
                            max_abs_e=float(np.max(np.abs(states[:, 0]))),
                            kappa=float(kappa), states=states.tolist()))
    return records


def aggregate(records):
    summary = summarise(records)
    if 'cp_coverage' in records[0]:
        summary['cp_coverage'] = float(np.mean([r['cp_coverage'] for r in records]))
        summary['mean_residual'] = float(np.mean([r['mean_residual'] for r in records]))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--cp', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--residual', action='store_true', help='actor path is a residual checkpoint')
    parser.add_argument('--patch-observer', action='store_true', help='observer is a patch-token checkpoint')
    parser.add_argument('--dino-repo', type=Path)
    parser.add_argument('--dino-weights', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'choose a new output path: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    actor = load_residual(args.actor, args.device)[0] if args.residual else load_actor(args.actor, args.device)
    if args.patch_observer:
        observer = PatchImageObserver(args.observer, args.device, args.dino_repo, args.dino_weights)
    else:
        observer = ImageObserver(args.observer, args.device)
    cp = json.loads(args.cp.read_text())
    radius = float(cp['radius_normalized'])
    scenes = json.loads((args.data_dir / 'scenes.json').read_text())
    test_scenes = [s for s in scenes if s['split'] == 'test']
    if not test_scenes:
        raise ValueError('no test scenes in dataset')
    renderer = RoadRenderer(192, 192)
    nominal = {}
    for scene in test_scenes:
        nominal[f"scene_{scene['scene_id']}"] = dict(
            scene=scene,
            image=aggregate(image_rollout(actor, observer, renderer, scene, radius,
                                          episodes=args.episodes, seed=11)),
            oracle=aggregate(semantic_rollout(actor, scene, episodes=args.episodes, seed=11)))
    stress_scenes = make_scenes(9, seed=99, randomized=True)
    stress = {}
    for scene in stress_scenes:
        stress[f"scene_{scene['scene_id']}"] = dict(
            scene=scene,
            image=aggregate(image_rollout(actor, observer, renderer, scene, radius,
                                          episodes=max(3, args.episodes // 2), seed=23)))
    result = dict(actor=str(args.actor), observer=str(args.observer), cp=str(args.cp),
                  cp_radius_normalized=radius, nominal=nominal, stress=stress,
                  scope='rendered nominal scenes for nominal; randomized scenes are OOD stress, not CP coverage')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(nominal={k: {a: v['image'][a] for a in ('success', 'unsafe', 'mean_abs_e')}
                                 for k, v in nominal.items()},
                          stress={k: {a: v['image'][a] for a in ('success', 'unsafe', 'mean_abs_e')}
                                  for k, v in stress.items()}), indent=2))


if __name__ == '__main__':
    main()
