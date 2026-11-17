"""Deterministic actor export/load for verification and evaluation.

Stable-Baselines3's PPO policy is split into the MLP feature extractor and the
linear action head.  Verification needs exactly the deterministic mean action
(no sampling, no log-std), clipped to the normalized action box.
"""
import torch
from torch import nn


class DeterministicActor(nn.Module):
    def __init__(self, policy_net, action_net):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net

    def forward(self, observation):
        return self.action_net(self.policy_net(observation)).clamp(-1., 1.)


def export_actor(model, path):
    policy = model.policy
    actor = DeterministicActor(policy.mlp_extractor.policy_net, policy.action_net)
    torch.save(dict(format='lane_keeping_actor_v1',
                    policy_net=actor.policy_net.state_dict(),
                    action_net=actor.action_net.state_dict(),
                    observation_dim=int(policy.observation_space.shape[0]),
                    action_dim=int(policy.action_space.shape[0])), path)
    return actor


def load_actor(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') != 'lane_keeping_actor_v1':
        raise ValueError('unexpected lane-keeping actor checkpoint')
    # Rebuild the SB3 MLP shape from the saved Linear layers.
    linears = [(key, value) for key, value in saved['policy_net'].items() if key.endswith('.weight')]
    dims = [saved['observation_dim']] + [value.shape[0] for _, value in linears]
    layers = []
    for index in range(len(dims) - 1):
        layers.append(nn.Linear(dims[index], dims[index + 1]))
        # SB3 applies the activation after EVERY policy hidden layer,
        # including the last one before the separate action head.
        layers.append(nn.Tanh())
    policy_net = nn.Sequential(*layers)
    action_net = nn.Linear(dims[-1], saved['action_dim'])
    actor = DeterministicActor(policy_net, action_net)
    actor.policy_net.load_state_dict(saved['policy_net'])
    actor.action_net.load_state_dict(saved['action_net'])
    return actor.to(device).eval()
