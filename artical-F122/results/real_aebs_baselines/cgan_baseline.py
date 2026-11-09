"""Evaluate the original SPVC cGAN state_net on the real 640x640 test images.

The original cGAN is a state -> image generator; its observation model is the
state_net trained on generated 32x32 grayscale images. This script applies that
frozen state_net to real images downsampled to 32x32 grayscale, which is the
only way to compare it on the same real-image test set.
"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
from PIL import Image
from Combined_network.model import SubNet

S = 3.18272066116333

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--state-net', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    args = ap.parse_args()
    recs = json.load(open(args.manifest))['records']
    recs = [r for r in recs if r['split'] in ('validation', 'test')]
    net = SubNet([1024, 256, 64, 1])
    net.load_state_dict(torch.load(args.state_net, map_location='cpu', weights_only=True))
    net.eval()
    preds = []
    for i in range(0, len(recs), args.batch_size):
        batch = recs[i:i + args.batch_size]
        imgs = []
        for r in batch:
            im = Image.open(Path(args.data_root) / r['filename']).convert('L').resize((32, 32), Image.Resampling.BICUBIC)
            a = np.asarray(im, dtype=np.float32) / 255.
            a = 2. * a - 1.
            imgs.append(torch.from_numpy(a).reshape(-1))
        with torch.no_grad():
            p = net(torch.stack(imgs)).reshape(-1).numpy() * S
        preds.append(p)
        if (i // args.batch_size) % 20 == 0:
            print('processed', min(i + args.batch_size, len(recs)), '/', len(recs), flush=True)
    preds = np.concatenate(preds)
    split = np.array([r['split'] for r in recs])
    y = np.array([r['distance_m'] for r in recs], dtype=np.float64)
    weather = np.array([r['weather'] for r in recs])
    color = np.array([r['color'] for r in recs])
    def stats(mask):
        e = preds[mask] - y[mask]
        return dict(n=int(mask.sum()), mae_m=float(np.abs(e).mean()), rmse_m=float(np.sqrt((e ** 2).mean())),
                    p95_abs_m=float(np.quantile(np.abs(e), .95)), max_abs_m=float(np.abs(e).max()),
                    bias_m=float(e.mean()))
    cal = split == 'validation'; test = split == 'test'
    cal_res = np.abs(preds[cal] - y[cal]) / S
    n = len(cal_res); rank = min(n, max(1, int(math.ceil((n + 1) * .95))))
    q = float(np.sort(cal_res)[rank - 1])
    test_res = np.abs(preds[test] - y[test]) / S
    by_weather = {}
    for w in sorted(set(weather[test])):
        m = test & (weather == w)
        e = preds[m] - y[m]
        by_weather[w] = dict(n=int(m.sum()), mae_m=float(np.abs(e).mean()), p95_abs_m=float(np.quantile(np.abs(e), .95)))
    out = dict(method='original SPVC cGAN + state_net on real 32x32 grayscale',
               state_net=args.state_net, input='real RGB -> 32x32 grayscale -> [-1,1] -> state_net',
               calibration=stats(cal), test=stats(test),
               cp=dict(q_norm=q, radius_m=q * S, cal_coverage=float((cal_res <= q).mean()),
                       test_coverage=float((test_res <= q).mean()), n=n, rank=rank),
               by_weather_test=by_weather,
               limitation='cGAN/state_net was trained on the old 400-image generated-image library; this is the original front-end applied to the new real-image test set, not a same-budget retrain')
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
