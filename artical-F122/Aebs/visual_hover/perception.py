"""Render a downward camera and fit the three-output DINO projection and C."""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..curve_braking.models import Proxy
from ..curve_braking.vision import Encoder
from ..curve_braking.evaluate import save
from . import plant


def render(position, light, size=144):
    x, y, dz = position * plant.SCALE
    height = plant.HEIGHT + dz
    if height <= 0:
        raise ValueError('camera below ground')
    focal = 117 * size / 144
    gain = (0.85, 1., 1.12)[int(light)]

    def color(rgb):
        return tuple(min(255, int(c * gain)) for c in rgb)

    def pixel(a, b):
        return (size / 2 + focal * (a - x) / height,
                size / 2 - focal * (b - y) / height)

    image = Image.new('RGB', (size, size), color((77, 90, 83)))
    draw = ImageDraw.Draw(image)
    for k in range(-6, 7):
        d = k * .45
        draw.line([pixel(d, -3), pixel(d, 3)], fill=color((99, 107, 99)))
        draw.line([pixel(-3, d), pixel(3, d)], fill=color((99, 107, 99)))

    def rectangle(left, bottom, right, top, rgb):
        draw.rectangle([pixel(left, top), pixel(right, bottom)], fill=color(rgb))

    rectangle(-.53, -.53, .53, .53, (222, 218, 184))
    rectangle(-.48, -.48, .48, .48, (47, 65, 77))
    for a in [-.26, .16]:
        rectangle(a, -.26, a + .10, .26, (229, 224, 196))
    rectangle(-.17, -.045, .17, .045, (229, 224, 196))
    rectangle(-.45, .35, -.32, .45, (226, 144, 73))
    return np.asarray(image)


def static_data():
    rng = np.random.default_rng(170631)
    images, labels, lights = [], [], []
    seen = set()
    while len(images) < 2048:
        x = rng.uniform(-1.15, 1.15, 3)
        light = rng.integers(3)
        image = render(x, light)
        fingerprint = hashlib.sha256(image.tobytes()).digest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            labels.append(x)
            images.append(image)
            lights.append(light)
    return np.stack(images), np.array(labels), np.array(lights)


def fit(root, encoder, batch):
    torch.set_num_threads(3)
    torch.manual_seed(170632)
    images, labels, lights = static_data()
    split = np.repeat(['train', 'val', 'test'], [1536, 256, 256])
    np.savez_compressed(root/'static.npz', images=images, position=labels, light=lights, split=split)
    x = torch.tensor(labels, dtype=torch.float32)
    features = torch.cat([encoder(images[i:i+batch]) for i in range(0, len(x), batch)])
    train, val, test = (torch.from_numpy(np.flatnonzero(split == k)) for k in ['train', 'val', 'test'])
    mean = features[train].mean(0)
    _, _, basis = torch.pca_lowrank(features[train]-mean, q=160, center=False, niter=4)
    scores = (features - mean) @ basis
    std = scores[train].std(0).clamp_min(.001)
    scores /= std
    model = Proxy('linear', scores.shape[1])
    design = torch.column_stack((scores[train], torch.ones(len(train))))
    solution = torch.linalg.solve(design.T @ design + 2 * torch.eye(design.shape[1]), design.T @ x[train])
    with torch.no_grad():
        model.head.weight.copy_(solution[:-1].T)
        model.head.bias.copy_(solution[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00018)
    best = None
    for epoch in range(35):
        for ids in train[torch.randperm(len(train))].split(96):
            latent = model.head(scores[ids])
            loss = (latent-model.nominal(x[ids])).norm(dim=1).mean() + .1*(model.decode(latent)-x[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mse = float((model.decode(model.head(scores[val]))-x[val]).square().mean())
            if torch.linalg.cond(model.matrix()) < 10 and (best is None or mse < best[0]):
                best = mse, epoch+1, copy.deepcopy(model.state_dict())
    if best is None:
        raise RuntimeError('no valid projection on development split')
    model.load_state_dict(best[2])
    torch.save(dict(mean=mean, basis=basis, std=std, model=model.state_dict()), root/'frontend.pt')
    with torch.no_grad():
        err = model.decode(model.head(scores[test]))-x[test]
        save(root/'static.json', dict(seed=170631, train=len(train), val=len(val), test=len(test),
             pixel_distinct=len(images), best_epoch=best[1], validation_mse=best[0],
             physical_test_mae_m=(err.abs().mean(0).numpy()*plant.SCALE).tolist(), C=model.matrix().tolist()))


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('output', type=Path)
    cli.add_argument('--repo', default='../external/dinov2')
    cli.add_argument('--weights', default='../external/dinov2_weights/dinov2_vits14_pretrain.pth')
    cli.add_argument('--device', default='cuda:0')
    cli.add_argument('--batch', type=int, default=8)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fit(args.output, Encoder(args.repo, args.weights, args.device), args.batch)
