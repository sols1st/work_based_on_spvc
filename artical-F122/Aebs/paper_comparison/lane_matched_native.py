"""Fit an SPVT camera adapter and run the original SPVT safety training."""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lane_matched_data import write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob('*.py'))}


def spectral_layer(layer):
    nn.init.orthogonal_(layer.weight)
    nn.init.zeros_(layer.bias)
    return nn.utils.spectral_norm(layer)


class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            spectral_layer(nn.Conv2d(3, 32, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_layer(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            nn.Flatten(), spectral_layer(nn.Linear(64 * 2 * 4, 1)))

    def forward(self, gray, state):
        labels = (state / state.new_tensor([80., 12.]))[:, :, None, None].expand(-1, -1, 8, 16)
        return self.net(torch.cat([gray.reshape(-1, 1, 8, 16), labels], dim=1))


def orthogonal_penalty(weights):
    penalty = 0
    for weight in weights:
        row = weight if weight.shape[0] <= weight.shape[1] else weight.T
        gram = row @ row.T
        penalty = penalty + (gram - torch.diag_embed(torch.diagonal(gram))).square().sum()
    return penalty


def pid_target(state):
    action = (-10. * (state[:, 0] / 100) - 3. * state[:, 1]).clamp(-30, 30)
    return ((action + 30) / 60 * 2 - 1)[:, None]


def setup(output, upstream):
    root = output / 'native'
    root.mkdir(exist_ok=False)
    for name in ('model', 'spvt'):
        shutil.copytree(upstream / name, root / name,
                        ignore=shutil.ignore_patterns('__pycache__', 'logs', 'pretrained_ckpts'))
    shutil.copy2(upstream / 'train.py', root)
    (root / 'data').mkdir()
    sys.path[:0] = [str(root.resolve()), str((root / 'spvt').resolve())]
    return root


def front(output, upstream, device):
    start = time.monotonic()
    root = setup(output, upstream)
    from model.model import get_model
    data = dict(np.load(output / 'bank/train.npz'))
    gray = torch.from_numpy(data['gray']).to(device).float() / 127.5 - 1
    state = torch.from_numpy(data['states_cm_degrees']).to(device)
    torch.save(state.cpu(), root / 'data/carla_states.pt')
    model = get_model().to(device)
    g = model.generator
    for layer in g.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
    d = ConditionalDiscriminator().to(device)
    go = torch.optim.Adam(g.parameters(), lr=.0007, betas=(.5, .999))
    do = torch.optim.Adam(d.parameters(), lr=.0007, betas=(.5, .999))
    weights = [layer.weight for layer in g.modules() if isinstance(layer, nn.Linear)]
    history = []
    for epoch in range(1, 101):
        for ids in torch.randperm(len(state), device=device).split(128):
            x, real = state[ids], gray[ids]
            z = x.new_empty(len(x), 10).uniform_(-1., 1.)
            fake = g(z, x)
            d_loss = F.relu(1 - d(real, x)).mean() + F.relu(1 + d(fake.detach(), x)).mean()
            do.zero_grad(set_to_none=True)
            d_loss.backward()
            do.step()
            g_loss = -d(fake, x).mean() + .0001 * orthogonal_penalty(weights)
            go.zero_grad(set_to_none=True)
            g_loss.backward()
            go.step()
        if epoch % 10 == 0:
            row = dict(stage='generator', epoch=epoch, loss_G=float(g_loss), loss_D=float(d_loss))
            history.append(row)
            print(row, flush=True)
    torch.save(dict(generator_state_dict=g.state_dict()), root / 'generator.pth')
    actor = model.controller
    target = pid_target(state)
    opt = torch.optim.Adam(actor.parameters(), lr=.0005)
    for epoch in range(1, 201):
        for ids in torch.randperm(len(state), device=device).split(256):
            loss = F.mse_loss(actor(gray[ids]), target[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 20 == 0:
            row = dict(stage='camera_PID_imitation', epoch=epoch, loss=float(loss))
            history.append(row)
            print(row, flush=True)
    torch.save(actor.state_dict(), root / 'anchor.pth')
    write_json(root / 'front_training.json', dict(
        seconds=time.monotonic()-start, generator='author MLPGenerator with fresh orthogonal weights',
        discriminator='conditioned convolutional discriminator with spectral normalization, hinge loss',
        unspecified_paper_choices='Adam betas .5/.999; orthogonal penalty weight 1e-4; discriminator conv widths 32 and 64',
        generator_epochs=100, generator_batch=128, generator_lr=.0007, latent_train=[-1,1],
        anchor='author E2ENetworkBN, supervised on common target grayscale images and original calculate_action PID labels',
        anchor_epochs=200, anchor_batch=256, anchor_lr=.0005, history=history,
        bank_sha256=sha(output/'bank/train.npz'),
        generator_sha256=sha(root/'generator.pth'), anchor_sha256=sha(root/'anchor.pth'),
        original_sources=sources(root)))


def train(output, epochs, device):
    root = (output / 'native').resolve()
    front_log = json.loads((root/'front_training.json').read_text())
    assert front_log['original_sources'] == sources(root)
    os.chdir(root)
    sys.path[:0] = [str(root), str(root/'spvt')]
    sys.argv = [str(root/'train.py'), '--num_epochs', str(epochs), '--batch_size', '256',
                '--bound_method', 'IBP', '--seed', '261', '--device', device,
                '--gen_ckpt', str(root/'generator.pth'), '--control_ckpt', str(root/'anchor.pth'),
                '--run_id', 'logs/matched', '--tb_path', 'logs/matched/runs', '--ckpt_path', 'logs/matched/ckpts']
    from spvt.args import get_args
    from train import train as upstream_train
    config = get_args()
    start = time.monotonic()
    args_dict = copy.deepcopy(vars(config))
    upstream_train(config)
    assert front_log['original_sources'] == sources(root), 'native source changed'
    checks = sorted((root/'spvt/logs/matched/ckpts').glob('*.pth'))
    assert (root/f'spvt/logs/matched/ckpts/checkpoint_epoch_{epochs}.pth').exists()
    write_json(root/'safety_training.json', dict(seconds=time.monotonic()-start, settings=args_dict,
        checkpoints={f.name:sha(f) for f in checks}, source_unchanged=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['front','train'])
    p.add_argument('output', type=Path)
    p.add_argument('--upstream', type=Path, default=Path('/root/SPVT'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    random.seed(261)
    np.random.seed(261)
    torch.manual_seed(261)
    torch.set_num_threads(a.threads)
    if a.phase == 'front':
        front(a.output, a.upstream, a.device)
    else:
        train(a.output, a.epochs, a.device)
