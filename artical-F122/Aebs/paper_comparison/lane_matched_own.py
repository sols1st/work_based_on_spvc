"""Train a fresh RGB observation head and robust lane controller on shared states."""
import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.lane_keeping.learned_sbc import save_actor
from Aebs.lane_keeping.policy import DeterministicActor
from .lane_matched_data import write_json
from .lane_matched_native import pid_target, sha

REPO = Path('../external/dinov2').resolve()
WEIGHTS = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth').resolve()


def state_normalized(states, device):
    state = torch.tensor(states, device=device)
    return torch.stack([state[:, 0] / 85, torch.deg2rad(state[:, 1]) / .30], 1)


def pool_tokens(tokens):
    patches = tokens.transpose(1, 2).reshape(-1, 384, 16, 16)
    return torch.nn.functional.avg_pool2d(patches, 4).flatten(1)


@torch.no_grad()
def features(backbone, rgb, device):
    parts = []
    for i in range(0, len(rgb), 64):
        image = torch.from_numpy(np.array(rgb[i:i + 64])).to(device).permute(0, 3, 1, 2).float() / 255
        tokens = backbone.backbone.forward_features(backbone.preprocess(image))['x_norm_patchtokens']
        parts.append(pool_tokens(tokens).cpu())
        if i and i % 4096 == 0:
            print('RGB patches', i, '/', len(rgb), flush=True)
    return torch.cat(parts)


def get_actions(actor, x):
    return actor(torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)).flatten()


def rollout_cost(actor, state, errors):
    total = state.new_zeros(len(state))
    previous = None
    for error in errors.unbind(1):
        u = get_actions(actor, state + error) * (math.pi / 6)
        e, heading = state.unbind(1)
        state = torch.stack([e + .25 / .85 * torch.sin(.30 * heading),
                             heading + .1 / .30 * torch.tan(u)], 1)
        total = total + (.85 * state[:, 0] / .10).square() + .2 * (.30 * state[:, 1] / .05).square()
        total = total + .01 * (u / .10).square()
        if previous is not None:
            total = total + .05 * ((u - previous) / .05).square()
        total = total + 25 * (state.abs() - state.new_tensor([.8 / .85, 1.])).relu().square().sum(1)
        previous = u
    return total / errors.shape[1] + (.85 * state[:, 0] / .10).square()


def fit(output, device):
    start = time.monotonic()
    torch.manual_seed(261)
    directory = output / 'own'
    directory.mkdir(exist_ok=False)
    bank = output / 'bank'
    train = np.load(bank/'train.npz')['states_cm_degrees']
    dev = np.load(bank/'development.npz')['states_cm_degrees']
    train_x, dev_x = state_normalized(train, device), state_normalized(dev, device)
    backbone = FrozenDinoV2(REPO, WEIGHTS, device)
    train_f = features(backbone, np.load(bank/'train_rgb.npy', mmap_mode='r'), device)
    dev_f = features(backbone, np.load(bank/'development_rgb.npy', mmap_mode='r'), device)
    # Standardize only with training data. Tune a small ridge grid on development.
    mean = train_f.mean(0)
    std = train_f.std(0).clamp_min(.01)
    train_f = (train_f - mean) / std
    dev_f = (dev_f - mean) / std
    target = train_x.cpu()
    bias = target.mean(0)
    gram = train_f.T @ train_f
    rhs = train_f.T @ (target - bias)
    diagonal = torch.eye(len(gram), dtype=gram.dtype)
    best = float('inf')
    for penalty in (1., 10., 100.):
        weight = torch.linalg.solve(gram + penalty * diagonal, rhs)
        errors = dev_f @ weight + bias - dev_x.cpu()
        mse = float(errors.square().mean())
        print(dict(ridge=penalty, dev_mse=mse), flush=True)
        if mse < best:
            best, best_penalty, best_weight = mse, penalty, weight
    observations = (train_f @ best_weight + bias).to(device)
    dev_errors = (dev_f @ best_weight + bias).to(device) - dev_x
    del train_f, dev_f, gram, rhs, diagonal, backbone
    torch.save(dict(format='lane_pooled_rgb_dino_vision_sls_v1', mean=mean, std=std,
                    weight=best_weight, bias=bias, weights_sha256=sha(WEIGHTS),
                    data_sha256=sha(bank/'train.npz'), ridge=best_penalty), directory/'patch_projection.pt')
    actor = DeterministicActor(nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()),
                               nn.Linear(64, 1)).to(device)
    labels = pid_target(torch.from_numpy(train).to(device)).flatten()
    optimizer = torch.optim.Adam(actor.parameters(), lr=.0005)
    for _ in range(75):
        for ids in torch.randperm(len(train_x), device=device).split(256):
            loss = (get_actions(actor, observations[ids]) - labels[ids]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    save_actor(directory/'actor_initial.pt', copy.deepcopy(actor).cpu())
    dev_q = float(torch.quantile(dev_errors.abs().amax(dim=1), .95))
    radius = max(.075, math.ceil((dev_q + .015) * 100) / 100)
    # Separate one-stage robust control training and final neural certificate.
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    history = []
    for step in range(1, 1201):
        ids = torch.randint(len(train_x), (128,), device=device)
        x = train_x[ids]
        errors = x.new_empty(len(x), 32, 2).uniform_(-radius, radius)
        for _ in range(3):
            errors.requires_grad_(True)
            worst = rollout_cost(actor, x, errors).mean()
            gradient, = torch.autograd.grad(worst, errors)
            errors = (errors.detach() + radius / 2 * gradient.sign()).clamp(-radius, radius)
        nominal = rollout_cost(actor, x, torch.zeros_like(errors)).mean()
        stressed = rollout_cost(actor, x, errors).mean()
        loss = .5 * (nominal + stressed)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if step % 100 == 0:
            row = dict(step=step, nominal=float(nominal), stressed=float(stressed))
            history.append(row)
            print(row, flush=True)
    save_actor(directory/'actor.pt', actor.cpu())
    write_json(directory/'training.json', dict(
        seconds=time.monotonic()-start, train_images=len(train), dev_images=len(dev),
        projection_ridge=best_penalty, projection_best_dev_mse=best,
        projection='4x4 average pooled 384D patch tokens, standardized ridge 6144->2, fixed Cx',
        actor='3->64 Tanh->64 Tanh->1 clipped normalized action, third input known zero curvature',
        anchor='75 epochs PID imitation on projected source images, using the same SPVT calculate_action labels',
        radius_dev_frame_q=dev_q, design_radius=radius,
        radius_rule='round up developer frame q95+.015 to centesimal, floor .075',
        refinement='1200 Adam steps, 32-step nominal and 3 projected gradient ascent steps for uncertain images, both terms weighted equally',
        training_cost='lateral and heading errors, actuator magnitude, actuator step change, unsafe rectangle penalty, terminal lateral error',
        barrier='trained separately from scratch after visual head and actor freeze',
        actor_sha256=sha(directory/'actor.pt'), projection_sha256=sha(directory/'patch_projection.pt'),
        history=history))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    fit(args.output, args.device)
