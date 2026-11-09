"""Train PPO on semantic or calibrated noisy lane observations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from .env import LaneKeepingEnv, NoisyLaneEnv, BoxErrorModel, rollout, summarise
from .policy import export_actor
from . import core


def make_env(mode, cp_path=None, kappa=None, seed=None, initial_low=None, initial_high=None):
    if mode == 'semantic':
        return LaneKeepingEnv(kappa=kappa, seed=seed, initial_low=initial_low, initial_high=initial_high)
    if mode == 'noisy':
        if cp_path is None:
            raise ValueError('noisy mode requires --cp')
        model = BoxErrorModel.from_dict(json.loads(Path(cp_path).read_text())['error_model'])
        return NoisyLaneEnv(model, kappa=kappa, seed=seed, initial_low=initial_low,
                            initial_high=initial_high)
    raise ValueError('mode must be semantic or noisy')


def evaluate(actor, mode, cp_path, episodes=100, seed=1234):
    # Evaluate over all three curvature values with independent initial states.
    results = {}
    for kappa in (-.01, 0., .01):
        env = make_env(mode, cp_path, kappa=kappa, seed=seed + int(1000 * (kappa + .01)))
        results[f'{kappa:+.3f}'] = summarise(rollout(actor, env, episodes=episodes, seed=seed))
    env = make_env(mode, cp_path, kappa=None, seed=seed + 17)
    results['random_kappa'] = summarise(rollout(actor, env, episodes=episodes, seed=seed + 17))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['semantic', 'noisy'], default='semantic')
    parser.add_argument('--cp', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--n-steps', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--train-e-half', type=float, default=None,
                        help='train from a wider initial box half-width (metres); evaluation still starts near centre')
    parser.add_argument('--train-theta-half', type=float, default=None,
                        help='train from a wider initial heading half-width (radians)')
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    set_random_seed(args.seed)
    initial_low = initial_high = None
    if args.train_e_half is not None or args.train_theta_half is not None:
        e_half = core.INIT_E if args.train_e_half is None else args.train_e_half
        t_half = core.INIT_THETA if args.train_theta_half is None else args.train_theta_half
        initial_low, initial_high = [-e_half, -t_half], [e_half, t_half]
    env = make_env(args.mode, args.cp, seed=args.seed, initial_low=initial_low,
                   initial_high=initial_high)
    model = PPO('MlpPolicy', env, seed=args.seed, verbose=1, n_steps=args.n_steps,
                batch_size=args.batch_size, learning_rate=args.learning_rate,
                gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=.0,
                policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
                device='cpu')
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.output_dir / 'ppo_model.zip')
    actor = export_actor(model, args.output_dir / 'actor.pt')
    evaluation = evaluate(actor, args.mode, args.cp, episodes=args.episodes, seed=args.seed + 101)
    report = dict(mode=args.mode, timesteps=args.timesteps, seed=args.seed,
                  cp=str(args.cp) if args.cp else None, evaluation=evaluation)
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
