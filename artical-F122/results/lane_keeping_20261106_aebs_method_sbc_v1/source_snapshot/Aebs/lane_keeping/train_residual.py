"""Train a bounded neural residual on top of the certified LQR controller.

The certificate covers any residual with ``|tanh| <= 1`` and the chosen
``eps``; training changes performance, not the certificate class.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from . import core
from .certify_linear import lqr
from .env import ResidualLaneEnv, LaneKeepingEnv, rollout, summarise
from .residual import ResidualActor, ResidualNet, save_residual, lqr_only


def export_residual(model, K, P, eps, metadata, path, flat_c=None):
    policy = model.policy
    net = ResidualNet(policy.mlp_extractor.policy_net[0].out_features)
    # SB3 policy_net is Linear/Tanh/Linear/Tanh; append the output tanh.
    state = {}
    for index, layer in enumerate(policy.mlp_extractor.policy_net):
        if isinstance(layer, torch.nn.Linear):
            state[f'net.{index}.weight'] = layer.weight.detach().clone()
            state[f'net.{index}.bias'] = layer.bias.detach().clone()
    state['net.4.weight'] = policy.action_net.weight.detach().clone()
    state['net.4.bias'] = policy.action_net.bias.detach().clone()
    net.load_state_dict(state)
    actor = ResidualActor(K, P, net, eps, flat_c)
    save_residual(path, actor, metadata)
    return actor


def evaluate(actor, episodes=50, seed=123):
    results = {}
    for kappa in (-.01, 0., .01):
        env = LaneKeepingEnv(kappa=kappa, seed=seed + int(1000 * (kappa + .01)))
        results[f'{kappa:+.3f}'] = summarise(rollout(actor, env, episodes=episodes, seed=seed))
    env = LaneKeepingEnv(kappa=None, seed=seed + 17)
    results['random_kappa'] = summarise(rollout(actor, env, episodes=episodes, seed=seed + 17))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--eps', type=float, default=.5)
    parser.add_argument('--flat-c', type=float, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--episodes', type=int, default=50)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    set_random_seed(args.seed)
    K, P, _ = lqr()
    env = ResidualLaneEnv(K, P, eps=args.eps, flat_c=args.flat_c, seed=args.seed)
    model = PPO('MlpPolicy', env, seed=args.seed, verbose=1, n_steps=1024,
                batch_size=256, learning_rate=3e-4, gamma=.99, gae_lambda=.95,
                policy_kwargs=dict(net_arch=[32, 32], activation_fn=torch.nn.Tanh),
                device='cpu')
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    metadata = dict(K=K.tolist(), P=P.tolist(), eps=args.eps, flat_c=args.flat_c, timesteps=args.timesteps,
                    seed=args.seed, certificate='residual_sbc.json',
                    residual_bound='eps * clamp(B(x),0,1) * tanh(z)')
    actor = export_residual(model, K, P, args.eps, metadata, args.output_dir / 'residual.pt',
                            flat_c=args.flat_c)
    model.save(args.output_dir / 'ppo_model.zip')
    lqr_actor = torch.nn.Identity()  # placeholder replaced below
    # Evaluate LQR by wrapping it as a callable actor.
    class LQRActor(torch.nn.Module):
        def forward(self, observation):
            return lqr_only(observation, K)
    lqr_eval = evaluate(LQRActor(), episodes=args.episodes, seed=args.seed + 101)
    residual_eval = evaluate(actor, episodes=args.episodes, seed=args.seed + 101)
    report = dict(eps=args.eps, timesteps=args.timesteps, seed=args.seed,
                  lqr=lqr_eval, residual=residual_eval)
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
