"""Build a frozen real-RGB AEBS artifact: DINO observation, PPO, CP contract.

Input is the existing real-RGB DINO CLS cache (384-D) from
carla_data_spvc_expanded/full01. This is the observation/PPO/CP stage; SBC
training and continuous verification are separate stages.
"""
import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from Aebs.vision_sls.model import ObservationProjection, VisionSLSPolicy
from Aebs.vision_sls.env import ObservationEnv
from Aebs.vision_sls.rollout import rollout, initial_grid
from Combined_network.model import CombinedPolicyNetwork

S = 3.18272066116333
ROOT = Path(__file__).resolve().parents[2]
DINO_REPO = ROOT.parent / 'external/dinov2'
DINO_WEIGHTS = ROOT.parent / 'external/dinov2_weights/dinov2_vits14_pretrain.pth'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def load_h5(path, split_name=None):
    with h5py.File(path, 'r') as f:
        d = {k: np.array(f[k]) for k in ['features', 'distance_m', 'distance_bin_025m', 'filename', 'weather', 'color']}
        d['split'] = np.array([split_name] * len(d['features'])) if split_name else dec(np.array(f['split']))
    for k in ['filename', 'weather', 'color', 'split']:
        d[k] = dec(d[k])
    d['bin'] = np.array(['bin_' + str(int(x)) for x in d['distance_bin_025m']])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default=str(ROOT/'results/dino_expanded_v1/02_dino_h5/features.h5'))
    ap.add_argument('--test-features', type=Path)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--ppo-steps', type=int, default=100000)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit('output exists')
    out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = load_h5(args.features)
    X, y, bins, split = d['features'].astype(np.float32), d['distance_m'].astype(np.float32), d['bin'], d['split']
    names, weather, color = d['filename'], d['weather'], d['color']
    if args.test_features:
        td = load_h5(args.test_features, 'test')
        X = np.concatenate([X, td['features'].astype(np.float32)])
        y = np.concatenate([y, td['distance_m'].astype(np.float32)])
        bins = np.concatenate([bins, td['bin']]); split = np.concatenate([split, td['split']])
        names = np.concatenate([names, td['filename']]); weather = np.concatenate([weather, td['weather']])
        color = np.concatenate([color, td['color']])
    tr = np.where(split == 'train')[0]
    cal = np.where(split == 'validation')[0]
    test = np.where(split == 'test')[0]
    if len(test) == 0:
        val_bins = sorted(set(bins[cal])); test = np.where(np.isin(bins, val_bins[1::2]))[0]; cal = np.where(np.isin(bins, val_bins[0::2]))[0]
    train_bins = sorted(set(bins[tr])); cal_bins = sorted(set(bins[cal])); test_bins = sorted(set(bins[test]))
    # Observation: train only; calibration/test bins are held out.
    mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
    xt = torch.from_numpy(X[tr]); yt = torch.from_numpy(y[tr] / S)
    model = ObservationProjection(384, 128)
    model.feature_mean.copy_(torch.from_numpy(mu)[None]); model.feature_std.copy_(torch.from_numpy(sd)[None])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = (1e9, None, 0)
    idx = np.arange(len(tr)); rng = np.random.default_rng(args.seed); rng.shuffle(idx)
    nval = max(500, int(.1 * len(idx))); vsel = idx[:nval]; fsel = idx[nval:]
    for ep in range(1, args.epochs + 1):
        model.train(); order = torch.randperm(len(fsel))
        for b in order.split(64):
            j = torch.from_numpy(fsel[b.numpy()])
            loss = (model(xt[j]) - yt[j, None]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                pv = model(xt[torch.from_numpy(vsel)]); mae = float((pv - yt[torch.from_numpy(vsel), None]).abs().mean() * S)
            if mae < best[0]:
                best = (mae, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1]); model.eval().requires_grad_(False)
    with torch.no_grad():
        obs = model(torch.from_numpy(X)).numpy().reshape(-1)
    weights_hash = sha256(DINO_WEIGHTS)
    torch.save(dict(format='vision_sls_aebs_v1', projection=model.state_dict(), feature_dim=384,
                    hidden_dim=128, distance_scale_m=S, dino_repo=str(DINO_REPO.resolve()),
                    dino_weights=str(DINO_WEIGHTS.resolve()), best_epoch=best[2], seed=args.seed,
                    provenance=dict(data='real 640x640 RGB; DINO CLS 384 cache', dino_model='dinov2_vits14',
                                    dino_weights_sha256=weights_hash)), out/'observation.pt')
    np.savez(out/'observations.npz', distance_m=y, observation=obs, split=split, bin=bins,
             filename=names, weather=weather, color=color)
    env = ObservationEnv(S, y[tr], obs[tr], mixed=True)
    ppo = PPO('MlpPolicy', env, n_steps=256, batch_size=64, seed=args.seed, device='cpu', verbose=0)
    print(f'PPO {args.ppo_steps} steps; train {len(tr)}; cal {len(cal)}; test {len(test)}', flush=True)
    ppo.learn(total_timesteps=args.ppo_steps)
    ppo.save(out/'ppo_initial.zip')
    policy = VisionSLSPolicy(CombinedPolicyNetwork(ppo.policy.mlp_extractor.policy_net, ppo.policy.action_net), S).eval()
    policy.set_state_distance_scale(S)
    torch.save(dict(format='vision_sls_policy_v1', state_dict=policy.state_dict(),
                    observation_sha256=sha256(out/'observation.pt'), metadata=policy.metadata(),
                    actor_hidden=[64, 64], activation='Tanh'), out/'policy.pt')
    cal_res = np.abs(obs[cal] - y[cal] / S)
    n = len(cal_res); rank = min(n, max(1, int(np.ceil((n + 1) * .95))))
    q = float(np.sort(cal_res)[rank - 1])
    test_res = np.abs(obs[test] - y[test] / S)
    contract = dict(format='vision_sls_state_cp_reference_v1', coefficients=[S, S, S, S],
                    envelope_units='m',score_units='normalized_distance',distance_scale_m=S,
                    fit=dict(degree=3, floor_m=.01, note='constant metre scale S; physical radius is q*S'),
                    calibration=dict(q_hat=q, rank=rank, n=n, alpha=.05, marginal_target=.95),
                    coverage=dict(calibration=dict(samples=n, covered=int((cal_res <= q).sum()),
                                                   mean_radius_m=float(q * S), max_error_m=float(cal_res.max() * S)),
                                  test=dict(samples=len(test), covered=int((test_res <= q).sum()),
                                            fraction=float((test_res <= q).mean()), max_error_m=float(test_res.max() * S))),
                    source_observation_sha256=sha256(out/'observation.pt'),
                    source_policy_sha256=sha256(out/'policy.pt'),
                    input='true distance used offline for calibration/replay; deployment uses image only',
                    statistical_status='finite real-image library; calibration bins and test bins disjoint at 0.25m-bin level',
                    independent_trajectory_calibration=False, formal_image_certificate=False)
    (out/'perception_contract.json').write_text(json.dumps(contract, indent=2))
    def slim(r):
        return dict(episodes=r['episodes'], counts=r['counts'], lookup_max_error_m=r['lookup_max_error_m'], mode=r['mode'])
    all_d, all_o = y, obs
    behavior = dict(nominal=slim(rollout(policy, S, initial_grid(20))),
                    train_images=slim(rollout(policy, S, initial_grid(20), y[tr], obs[tr])),
                    calibration_images=slim(rollout(policy, S, initial_grid(20), y[cal], obs[cal])),
                    test_images_dense=slim(rollout(policy, S, initial_grid(20), all_d, all_o)),
                    test_images_only=slim(rollout(policy, S, initial_grid(20), y[test], obs[test])))
    def mae_m(mask):
        return float(np.abs(obs[mask] * S - y[mask]).mean())
    metrics = dict(status='real_rgb_artifact_complete', scope='observation+PPO+CP; SBC not trained; no certificate',
                   data=dict(images=len(y), train=len(tr), calibration=len(cal), test=len(test),
                             train_bins=train_bins, calibration_bins=cal_bins, test_bins=test_bins),
                   observation=dict(feature_dim=384, hidden=128, best_epoch=best[2],
                                    train_mae_m=mae_m(tr), cal_mae_m=mae_m(cal), test_mae_m=mae_m(test),
                                    cal_max_m=float(np.abs(obs[cal] * S - y[cal]).max()),
                                    test_max_m=float(np.abs(obs[test] * S - y[test]).max())),
                   cp=dict(q_norm=q, radius_m=q * S, cal_coverage=float((cal_res <= q).mean()),
                           test_coverage=float((test_res <= q).mean()), n=n, rank=rank),
                   behavior=behavior, config=dict(ppo_steps=args.ppo_steps, epochs=args.epochs, seed=args.seed,
                                                  features=str(Path(args.features).resolve()),
                                                  test_features=str(args.test_features) if args.test_features else None))
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics[k] for k in ['observation', 'cp', 'behavior']}, indent=2, default=str))


if __name__ == '__main__':
    main()
