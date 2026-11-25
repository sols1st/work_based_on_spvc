"""Train a fresh RGB-DINO AEBS projection and cache clean or shifted predictions."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection
from .aebs_matched_data import write_json
from .aebs_matched_native import sha

CONDITIONS = (
    ('clean', 'RGB 原图'), ('dark', '亮度×0.7'), ('bright', '亮度×1.3'),
    ('contrast_low', '对比度×0.7'), ('contrast_high', '对比度×1.3'),
    ('blur10', '模糊 σ=10 源像素'), ('blur20', '模糊 σ=20 源像素'),
)


def transform_image(image, condition):
    if condition == 'clean':
        return image
    if condition in ('dark', 'bright'):
        return ImageEnhance.Brightness(image).enhance(.7 if condition == 'dark' else 1.3)
    if condition in ('contrast_low', 'contrast_high'):
        return ImageEnhance.Contrast(image).enhance(.7 if condition == 'contrast_low' else 1.3)
    if condition in ('blur10', 'blur20'):
        return image.filter(ImageFilter.GaussianBlur(10 if condition == 'blur10' else 20))
    raise ValueError(f'unknown condition {condition}')


class CarImages(Dataset):
    def __init__(self, source, files, condition, include_gray=False):
        self.source = source
        self.files = files
        self.condition = condition
        self.include_gray = include_gray

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.source / self.files[index]) as source:
            rgb = transform_image(source.convert('RGB'), self.condition)
            tensor = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)
            if not self.include_gray:
                return tensor
            gray = rgb.convert('L').resize((32, 32), Image.Resampling.BICUBIC)
            return tensor, torch.from_numpy(np.array(gray)).flatten()


def batches(source, files, condition, workers, include_gray=False):
    return DataLoader(CarImages(source, files, condition, include_gray),
                      batch_size=48, num_workers=workers, pin_memory=True,
                      multiprocessing_context='spawn' if workers else None)


def extract(backbone, source, files, condition, device, workers):
    parts = []
    for step, rgb in enumerate(batches(source, files, condition, workers), 1):
        parts.append(backbone(rgb.to(device, non_blocking=True).float() / 255.).cpu())
        if step % 50 == 0:
            print(f'{condition} features {min(step * 48, len(files))}/{len(files)}', flush=True)
    return torch.cat(parts)


def fit(experiment, workers, epochs, device):
    output = experiment / 'own_initial'
    output.mkdir(exist_ok=False)
    protocol = json.loads((experiment / 'data_protocol.json').read_text())
    data = dict(np.load(experiment / 'manifest.npz'))
    mask = np.isin(data['split'], ('train', 'development'))
    rows = np.flatnonzero(mask)
    train = torch.from_numpy(np.flatnonzero(data['split'][rows] == 'train')).to(device)
    dev = torch.from_numpy(np.flatnonzero(data['split'][rows] == 'development')).to(device)
    source = Path(protocol['source'])
    repo = Path('../external/dinov2').resolve()
    weights = Path('../external/dinov2_weights/dinov2_vits14_pretrain.pth').resolve()
    backbone = FrozenDinoV2(repo, weights, device)
    features = extract(backbone, source, data['files'][rows], 'clean', device, workers).to(device)
    distance = data['distance_m'][rows]
    scale = protocol['train_distance_scale_m']
    target = torch.tensor(distance / scale, dtype=torch.float32, device=device)[:, None]
    torch.manual_seed(251)
    head = ObservationProjection().to(device)
    head.feature_mean.copy_(features[train].mean(0, keepdim=True))
    head.feature_std.copy_(features[train].std(0, keepdim=True).clamp_min(.01))
    opt = torch.optim.Adam(head.parameters(), lr=.001)
    best_mse = float('inf')
    history = []
    for epoch in range(1, epochs + 1):
        for ids in train[torch.randperm(len(train), device=device)].split(64):
            loss = (head(features[ids]) - target[ids]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            errors = head(features[dev]) - target[dev]
            mse = float(errors.square().mean())
            mae_m = float(errors.abs().mean()) * scale
        history.append(dict(epoch=epoch, normalized_mse=mse, mae_m=mae_m))
        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            best_state = copy.deepcopy(head.state_dict())
        if epoch % 10 == 0:
            print(history[-1], flush=True)
    head.load_state_dict(best_state)
    with torch.no_grad():
        pred = head(features).flatten().cpu().numpy()
    torch.save(dict(format='vision_sls_aebs_v1', projection=head.cpu().state_dict(),
                    feature_dim=384, hidden_dim=128, distance_scale_m=scale,
                    dino_repo=str(repo), dino_weights=str(weights), best_epoch=best_epoch,
                    provenance=dict(dino_weights_sha256=sha(weights),
                                    data_manifest_sha256=sha(experiment / 'manifest.npz'),
                                    preprocessing='raw RGB -> bicubic224 -> ImageNet; static fit train images only')),
               output / 'observation.pt')
    np.savez_compressed(output / 'observations.npz', distance_m=distance, observation=pred,
                        split=data['split'][rows], appearance=data['appearance'][rows], files=data['files'][rows])
    np.savez(output / 'splits.npz', train=train.cpu().numpy())
    write_json(output / 'projection_training.json', dict(best_epoch=best_epoch,
               epochs=epochs, best_dev_mse=best_mse, history=history))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    fit(args.experiment, args.workers, args.epochs, args.device)


if __name__ == '__main__':
    main()
