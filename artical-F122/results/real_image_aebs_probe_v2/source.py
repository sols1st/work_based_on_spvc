"""Real-RGB CARLA AEBS distance probe (1-D visual observation + split CP).

This is a focused feasibility probe, not the full AEBS/SBC certificate:
- reads the frozen DINOv2 CLS features cached from the 24k real 640x640 RGB images;
- trains a small 384->128->64->1 distance head on scene-group-disjoint splits;
- reports distance error and split-conformal coverage.

The cache is used only to avoid re-encoding 19k PNGs; features were produced from
real RGB by the existing cache_expanded_features entry point.
"""
import argparse, json, math, os, shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn

torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))

S = 3.18272066116333  # AEBS distance scale, metres


def load_npz_h5(path):
    import h5py
    def dec(a):
        return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])
    with h5py.File(path, 'r') as f:
        d = {k: np.array(f[k]) for k in ['features', 'distance_m', 'distance_bin_025m', 'group_id', 'filename', 'weather', 'color', 'split']}
    for k in ['group_id', 'filename', 'weather', 'color', 'split']:
        d[k] = dec(d[k])
    return d


def stratified_cap(idx, groups, n, rng):
    """Cap total n while keeping roughly equal counts per group."""
    idx = np.asarray(idx)
    gs = np.asarray(groups)[idx]
    uniq = np.unique(gs)
    per = max(1, int(math.ceil(n / len(uniq))))
    out = []
    for g in uniq:
        take = idx[gs == g]
        if len(take) > per:
            take = rng.choice(take, per, replace=False)
        out.append(take)
    out = np.concatenate(out)
    if len(out) > n:
        out = rng.choice(out, n, replace=False)
    return np.sort(out)


class Head(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--train-samples', type=int, default=8000)
    ap.add_argument('--cal-samples', type=int, default=1500)
    ap.add_argument('--test-samples', type=int, default=1500)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f'output exists: {out}')
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    d = load_npz_h5(args.features)
    X = d['features'].astype(np.float32)
    y = d['distance_m'].astype(np.float32)
    groups = np.array(['bin_' + str(int(x)) for x in d['distance_bin_025m']])
    fine_groups = d['group_id'].astype(str)
    split = d['split'].astype(str)
    train_all = np.where(split == 'train')[0]
    val_all = np.where(split == 'validation')[0]
    train_groups = sorted(set(groups[train_all].tolist()))
    val_groups = sorted(set(groups[val_all].tolist()))
    # deterministic held-out validation groups: every other validation group
    cal_groups = val_groups[0::2]
    test_groups = val_groups[1::2]
    # keep 4 train groups for early stopping, disjoint from fit groups
    early_groups = train_groups[::7][:4]
    fit_groups = [g for g in train_groups if g not in set(early_groups)]

    def subset(groups_keep, n):
        idx = np.where(np.isin(groups, groups_keep))[0]
        return stratified_cap(idx, groups, n, rng)

    fit_idx = subset(fit_groups, args.train_samples)
    early_idx = subset(early_groups, 1000)
    cal_idx = subset(cal_groups, args.cal_samples)
    test_idx = subset(test_groups, args.test_samples)
    # split train (fit groups) 90/10 for early stopping too
    perm = rng.permutation(fit_idx)
    n_es = max(500, int(0.1 * len(perm)))
    es_idx, fit_idx = perm[:n_es], perm[n_es:]
    early_idx = np.concatenate([early_idx, es_idx])

    mu = X[fit_idx].mean(0)
    sd = X[fit_idx].std(0) + 1e-6
    def prep(idx):
        return torch.from_numpy((X[idx] - mu) / sd), torch.from_numpy(y[idx] / S)
    Xf, yf = prep(fit_idx)
    Xe, ye = prep(early_idx)
    Xc, yc = prep(cal_idx)
    Xt, yt = prep(test_idx)

    model = Head(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.MSELoss()
    best = (1e9, None, -1)
    bs = 256
    for ep in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(Xf))
        for i in range(0, len(order), bs):
            j = order[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xf[j]), yf[j])
            loss.backward()
            opt.step()
        sched.step()
        if ep % 25 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pe = model(Xe)
                mae = float(torch.mean(torch.abs(pe - ye)) * S)
            if mae < best[0]:
                best = (mae, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
            print(f'epoch {ep} early_mae_m={mae:.4f} best={best[0]:.4f}', flush=True)
    model.load_state_dict(best[1])
    model.eval()

    def predict(idx):
        with torch.no_grad():
            return model(torch.from_numpy((X[idx] - mu) / sd)).numpy() * S

    res = {}
    for name, idx in [('calibration', cal_idx), ('test', test_idx), ('fit', fit_idx)]:
        pred = predict(idx); true = y[idx]
        err = pred - true; ae = np.abs(err)
        res[name] = dict(n=int(len(idx)), mae_m=float(ae.mean()), rmse_m=float(np.sqrt((err**2).mean())),
                         p95_abs_m=float(np.quantile(ae, .95)), max_abs_m=float(ae.max()),
                         bias_m=float(err.mean()), groups=sorted(set(groups[idx].tolist())))
    # split CP on normalized absolute residual, finite-sample rank
    cal_ae = np.abs(predict(cal_idx) - y[cal_idx]) / S
    n = len(cal_ae)
    rank = int(math.ceil((n + 1) * .95))
    rank = min(max(rank, 1), n)
    q = float(np.sort(cal_ae)[rank - 1])
    test_ae = np.abs(predict(test_idx) - y[test_idx]) / S
    cp = dict(alpha=.05, n=n, rank=rank, q_norm=q, radius_m=q * S,
              cal_coverage=float(np.mean(cal_ae <= q)),
              test_coverage=float(np.mean(test_ae <= q)),
              test_miss=int(np.sum(test_ae > q)),
              method='split conformal, absolute normalized residual, one image = one score')
    # per weather/color on test
    by = {}
    for w in sorted(set(d['weather'][test_idx].tolist())):
        m = d['weather'][test_idx] == w
        ae = np.abs(predict(test_idx)[m] - y[test_idx][m])
        by[w] = dict(n=int(m.sum()), mae_m=float(ae.mean()), p95_abs_m=float(np.quantile(ae, .95)),
                     max_abs_m=float(ae.max()))
    cfg = dict(features=str(Path(args.features).resolve()), train_samples=len(fit_idx), early_samples=len(early_idx),
               calibration_samples=len(cal_idx), test_samples=len(test_idx), epochs=args.epochs, best_epoch=best[2],
               distance_scale_m=S, fit_groups=fit_groups, early_groups=early_groups,
               calibration_groups=cal_groups, test_groups=test_groups, seed=args.seed,
               model='384-128-64-1 GELU, standardized features', target='distance_m / 3.1827')
    metrics = dict(status='real_rgb_distance_probe_complete', scope='development probe; no SBC/PPO/closed-loop certificate',
                   data='real 640x640 RGB, frozen DINOv2 CLS 384 cache; no 32x32 grayscale',
                   config=cfg, results=res, cp=cp, by_weather_test=by,
                   all_groups=['train:' + ','.join(train_groups), 'cal:' + ','.join(cal_groups), 'test:' + ','.join(test_groups)])
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    torch.save({'state_dict': model.state_dict(), 'mu': mu, 'sd': sd, 'scale': S}, out / 'head.pt')
    # predictions
    with (out / 'predictions.csv').open('w') as f:
        f.write('filename,split,coarse_group,fine_group,weather,color,distance_m,pred_m,error_m\n')
        for split_name, idx in [('fit', fit_idx), ('calibration', cal_idx), ('test', test_idx)]:
            pred = predict(idx)
            for i, p in zip(idx, pred):
                f.write(f'{d["filename"][i]},{split_name},{groups[i]},{fine_groups[i]},{d["weather"][i]},"{d["color"][i]}",{y[i]:.6f},{p:.6f},{p-y[i]:.6f}\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
