"""Train the DINOv2 patch-token spatial readout for (e, theta)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..dino_latent.backbone import FrozenDinoV2
from . import core
from .data import load_dataset
from .patch_perception import PatchProjection, save_patch_checkpoint

ROOT = Path(__file__).resolve().parents[2]


def extract_patches(images, dino_repo, dino_weights, device, batch=16, cache=None):
    if cache is not None and Path(cache).is_file():
        saved = torch.load(cache, map_location='cpu', weights_only=False)
        if saved.get('images_shape') == list(images.shape):
            print(f'reusing patch cache {cache}', flush=True)
            return saved['patches']
    backbone = FrozenDinoV2(dino_repo, dino_weights, device)
    patches = []
    for start in range(0, len(images), batch):
        x = torch.from_numpy(images[start:start + batch]).to(device).float().permute(0, 3, 1, 2) / 255.
        x = backbone.preprocess(x)
        with torch.no_grad():
            tokens = backbone.backbone.forward_features(x)['x_norm_patchtokens']
        patches.append(tokens.cpu().half())
        if start % (batch * 8) == 0:
            print(f'patch features {min(start + batch, len(images))}/{len(images)}', flush=True)
    patches = torch.cat(patches)
    if cache is not None:
        torch.save(dict(images_shape=list(images.shape), patches=patches), cache)
    return patches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dino-repo', type=Path, default=ROOT.parent / 'external/dinov2')
    parser.add_argument('--dino-weights', type=Path,
                        default=ROOT.parent / 'external/dinov2_weights/dinov2_vits14_pretrain.pth')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--feature-cache', type=Path)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f'choose a new output directory: {args.output_dir}')
    args.output_dir.mkdir(parents=True)
    data = load_dataset(args.data_dir)
    states = data['states'].astype(np.float64)
    split = data['split']
    labels = np.stack([states[:, 0] / core.E_SCALE, states[:, 1] / core.THETA_SCALE], 1)
    patches = extract_patches(data['images'], args.dino_repo, args.dino_weights, args.device,
                              cache=args.feature_cache or args.output_dir / 'patches.pt')
    train = torch.as_tensor(np.where(split == 'train')[0], dtype=torch.long)
    val = torch.as_tensor(np.where(split == 'validation')[0], dtype=torch.long)
    test = torch.as_tensor(np.where(split == 'test')[0], dtype=torch.long)
    torch.manual_seed(args.seed)
    model = PatchProjection(384, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    target = torch.as_tensor(labels, dtype=torch.float32)
    best, best_state = float('inf'), None
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train))
        for start in range(0, len(train), args.batch):
            index = train[order[start:start + args.batch]]
            batch_patches = patches[index].float().to(args.device)
            prediction = model(batch_patches)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target[index.to('cpu')].to(args.device), beta=.02)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model(patches[val].float().to(args.device)).cpu()
            val_loss = float(torch.nn.functional.mse_loss(prediction, target[val]))
        if val_loss < best:
            best, best_state = val_loss, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print(f'epoch {epoch} val_mse {val_loss:.6g}', flush=True)
    model.load_state_dict(best_state)
    model.eval()

    def metrics(indices):
        with torch.no_grad():
            prediction = model(patches[indices].float().to(args.device)).cpu().numpy()
        error = prediction - labels[indices]
        mae = np.abs(error).mean(0) * np.array([core.E_SCALE, core.THETA_SCALE])
        rmse = np.sqrt((error ** 2).mean(0)) * np.array([core.E_SCALE, core.THETA_SCALE])
        return dict(n=int(len(indices)), mae_e=float(mae[0]), mae_theta=float(mae[1]),
                    rmse_e=float(rmse[0]), rmse_theta=float(rmse[1]),
                    max_abs_e=float(np.abs(error[:, 0]).max() * core.E_SCALE),
                    max_abs_theta=float(np.abs(error[:, 1]).max() * core.THETA_SCALE))
    report = dict(train=metrics(train), validation=metrics(val), test=metrics(test))
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2))
    save_patch_checkpoint(args.output_dir / 'patch_projection.pt', model, args.dino_repo,
                          args.dino_weights, 'dinov2_vits14',
                          dict(e=core.E_SCALE, theta=core.THETA_SCALE),
                          dict(data_dir=str(args.data_dir), epochs=args.epochs,
                               hidden=args.hidden, metrics=report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
