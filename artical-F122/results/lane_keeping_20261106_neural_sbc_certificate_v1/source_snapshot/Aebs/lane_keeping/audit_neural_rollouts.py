"""Independent deployment checks; samples are regression evidence, not proofs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from . import core
from .policy import load_actor
from .certify_neural import PolyhedralSBC
from .patch_perception import PatchImageObserver
from .renderer import RoadRenderer


def summary(states, actions, barrier):
    norm = states / np.array([core.E_SCALE, core.THETA_SCALE])
    with torch.no_grad():
        values = barrier(torch.tensor(norm)).numpy().squeeze(-1)
    return dict(episodes=states.shape[1], steps=states.shape[0]-1,
                unsafe_episodes=int((np.abs(norm)>1).any(axis=(0, 2)).sum()),
                max_abs_e=float(abs(states[..., 0]).max()),
                max_abs_theta=float(abs(states[..., 1]).max()),
                mean_abs_e=float(abs(states[1:, :, 0]).mean()),
                max_B=float(values.max()), max_B_drift=float(np.diff(values, axis=0).max()),
                max_action=float(abs(actions).max()))


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--source', type=Path, default=Path('results/lane_keeping_20261105'))
    p.add_argument('--dino-repo', type=Path, required=True)
    p.add_argument('--dino-weights', type=Path, required=True)
    args = p.parse_args(argv)
    if (args.package/'rollouts.json').exists():
        raise FileExistsError(args.package/'rollouts.json')
    torch.set_num_threads(4)
    actor = load_actor(args.package/'actor.pt')
    saved = torch.load(args.package/'sbc.pt', weights_only=False)
    barrier = PolyhedralSBC(saved['facets'], saved['plateau'], saved['scale'])
    rng = np.random.default_rng(20261106)
    records = {}
    for mode in ['uniform', 'greedy_corners', 'alternating']:
        n = 240
        x = core.sample_initial(rng, n)
        # Include all initial corners and zero explicitly.
        x[:5] = np.array([[e, t] for e in [-.25, .25] for t in [-.06, .06]]+[[0, 0]])
        states, actions = [x.copy()], []
        for t in range(core.HORIZON):
            # Arbitrarily switching registered curvatures is covered by a common B.
            ks = rng.choice(core.KAPPAS, size=n)
            normal = x / [.85, .3]
            if mode == 'greedy_corners':
                corners = np.array([[-.08, -.08], [-.08, .08], [.08, -.08], [.08, .08]])
                obs = np.concatenate([normal[:, None, :]+corners[None, :, :],
                                      np.broadcast_to((ks/.01)[:, None, None], (n, 4, 1))], 2)
                u = actor(torch.tensor(obs.reshape(-1, 3), dtype=torch.float32)).numpy().reshape(n, 4)*.45
                nx = core.dynamics_batch(np.repeat(x, 4, 0), u.flatten(), np.repeat(ks, 4)).reshape(n, 4, 2)
                score = ((nx/[.85, .3])@barrier.facets.numpy().T).max(2)
                selected = score.argmax(1)
                x = nx[np.arange(n), selected]
                action = u[np.arange(n), selected]
            else:
                err = rng.uniform(-.08, .08, (n, 2)) if mode == 'uniform' else .08*np.column_stack([
                    (-1.)**(np.arange(n)+t), (-1.)**(np.arange(n)+t//2)])
                obs = np.c_[normal+err, ks/.01]
                action = actor(torch.tensor(obs, dtype=torch.float32)).numpy().flatten()*.45
                x = core.dynamics_batch(x, action, ks)
            states.append(x.copy())
            actions.append(action.copy())
        states, actions = np.array(states), np.array(actions)
        records[mode] = summary(states, actions, barrier)
        np.savez_compressed(args.package/f'{mode}_trajectories.npz', states=states, actions=actions)
        print(json.dumps({mode: records[mode]}), flush=True)

    observer = PatchImageObserver(args.source/'patch_perception_v2/patch_projection.pt',
                                  'cpu', args.dino_repo, args.dino_weights)
    renderer = RoadRenderer(192, 192)
    scenes = json.loads((args.source/'patch_perception_v2/scenes.json').read_text())
    scenes = [s for s in scenes if s['split']=='test']
    scene_batch = [s for s in scenes for _ in range(2)]
    n = len(scene_batch)
    x = core.sample_initial(rng, n)
    ks = np.array([s['kappa'] for s in scene_batch])
    states, actions, errors, observations = [x.copy()], [], [], []
    for t in range(core.HORIZON):
        imgs = [renderer.render(x[i], ks[i], s['lighting'], s['occluder'],
                                s['camera_lateral'], s['camera_height'], s['blur'], s['noise'],
                                seed=s['scene_id']*100+t)
                for i, s in enumerate(scene_batch)]
        obs = observer(torch.from_numpy(np.array(imgs))).numpy()
        errors.append(obs-x/[.85, .3])
        observations.append(obs.copy())
        action = actor(torch.tensor(np.c_[obs, ks/.01], dtype=torch.float32)).numpy().flatten()*.45
        x = core.dynamics_batch(x, action, ks)
        states.append(x.copy())
        actions.append(action.copy())
        if t % 10 == 0:
            print(json.dumps(dict(image_step=t, episodes=n)), flush=True)
    states, actions, errors = np.array(states), np.array(actions), np.array(errors)
    image_result = summary(states, actions, barrier)
    norms = abs(errors).max(2)
    cp = json.loads((args.source/'cp/cp_patch_95.json').read_text())
    image_result.update(radius_used_by_certificate=.08,
                        fraction_frames_in_certificate_box=float((norms<=.08).mean()),
                        episodes_all_frames_in_certificate_box=int((norms<=.08).all(0).sum()),
                        fraction_frames_in_calibrated_box=float((norms<=cp['error_model']['radius']).mean()),
                        max_normalized_error=float(norms.max()),
                        per_episode_mean_abs_e=abs(states[1:, :, 0]).mean(0).tolist())
    records['rendered_images'] = image_result
    np.savez_compressed(args.package/'image_trajectories.npz', states=states, actions=actions,
                        errors=errors, observations=np.array(observations), kappas=ks)
    records['scope'] = 'Finite regression experiments, 80 steps, float32 deployed PPO. Image errors outside the certificate box are not certified by CP or by this conditional certificate.'
    (args.package/'rollouts.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(image_result), flush=True)


if __name__ == '__main__':
    main()
