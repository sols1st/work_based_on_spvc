"""Train and freeze the DINO -> (e, theta) projection on scene-group splits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..dino_latent.backbone import FrozenDinoV2
from . import core
from .data import load_dataset
from .perception import DINOProjection, save_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def extract_features(images, dino_repo, dino_weights, device, batch=16, cache=None):
    if cache is not None and Path(cache).is_file():
        cached = torch.load(cache, map_location='cpu', weights_only=False)
        if cached.get('images_shape') == list(images.shape):
            print(f'reusing feature cache {cache}', flush=True)
            return cached['features']
    backbone = FrozenDinoV2(dino_repo, dino_weights, device)
    features = []
    for start in range(0, len(images), batch):
        batch_images = torch.from_numpy(images[start:start + batch]).to(device)
        batch_images = batch_images.float().permute(0, 3, 1, 2) / 255.
        features.append(backbone(batch_images).cpu())
        if start % (batch * 4) == 0:
            print(f'DINO features {min(start + batch, len(images))}/{len(images)}', flush=True)
    features = torch.cat(features)
    if cache is not None:
        torch.save(dict(images_shape=list(images.shape), features=features), cache)
    return features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dino-repo', type=Path, default=ROOT.parent / 'external/dinov2')
    parser.add_argument('--dino-weights', type=Path,
                        default=ROOT.parent / 'external/dinov2_weights/dinov2_vits14_pretrain.pth')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--feature-cache', type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    data = load_dataset(args.data_dir)
    images = data['images']
    states = data['states'].astype(np.float64)
    split = data['split']
    labels = np.stack([states[:, 0] / core.E_SCALE, states[:, 1] / core.THETA_SCALE], axis=1)
    features = extract_features(images, args.dino_repo, args.dino_weights, args.device,
                                cache=args.feature_cache or args.output_dir / 'features.pt')
    train = np.where(split == 'train')[0]
    val = np.where(split == 'validation')[0]
    test = np.where(split == 'test')[0]
    if len(train) < 10 or len(val) < 5:
        raise ValueError('not enough train/validation scenes')

    torch.manual_seed(args.seed)
    model = DINOProjection(feature_dim=features.shape[1], hidden=args.hidden)
    model.feature_mean.copy_(features[train].mean(0, keepdim=True))
    model.feature_std.copy_(features[train].std(0, keepdim=True).clamp_min(.01))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    target = torch.as_tensor(labels, dtype=torch.float32)
    best, best_state = float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        for start in range(0, len(train), args.batch):
            batch = train[order[start:start + args.batch].numpy()]
            prediction = model(features[batch])
            # Smooth L1 avoids a few large residuals dominating the projection.
            loss = torch.nn.functional.smooth_l1_loss(prediction, target[batch], beta=.05)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(features[val])
            val_loss = float(torch.nn.functional.mse_loss(prediction, target[val]))
        if val_loss < best:
            best, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f'epoch {epoch} val_mse {val_loss:.6g}', flush=True)
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        def metrics(indices):
            prediction = model(features[indices]).numpy()
            error = prediction - labels[indices]
            mae = np.abs(error).mean(0) * np.array([core.E_SCALE, core.THETA_SCALE])
            rmse = np.sqrt((error ** 2).mean(0)) * np.array([core.E_SCALE, core.THETA_SCALE])
            return dict(n=int(len(indices)), mae_e=float(mae[0]), mae_theta=float(mae[1]),
                        rmse_e=float(rmse[0]), rmse_theta=float(rmse[1]),
                        max_abs_e=float(np.abs(error[:, 0]).max() * core.E_SCALE),
                        max_abs_theta=float(np.abs(error[:, 1]).max() * core.THETA_SCALE))
        report = dict(train=metrics(train), validation=metrics(val), test=metrics(test))
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    save_checkpoint(
        args.output_dir / 'projection.pt', model,
        model.feature_mean, model.feature_std, args.dino_repo, args.dino_weights,
        'dinov2_vits14', dict(e=core.E_SCALE, theta=core.THETA_SCALE),
        dict(data_dir=str(args.data_dir), train_scenes=int(len(train)),
             image_size=int(images.shape[1]), epochs=args.epochs, hidden=args.hidden,
             metrics=report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
