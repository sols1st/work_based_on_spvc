"""Dimension-flexible native representation audit, not a controller benchmark.

Each frozen representation receives NEW, equally trained distance probes and a
NEW physical-to-native-latent q. No candidate is scored through the R2 decoder
or actor. Probes are measurements only: deployment is not changed to 1-D.
VICReg is a feature-level same-distance-pair adaptation, not a full reproduction.
Historical R2 has a different training history; it is not a budget-matched arm.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256


class FeatureCodec(nn.Module):
    def __init__(self, dimension, variational=False):
        super().__init__()
        if not 1 <= dimension <= 384:
            raise ValueError("Invalid native dimension")
        self.variational = variational
        self.encoder = nn.Sequential(nn.Linear(384, 128), nn.GELU(),
                                     nn.Linear(128, dimension * (2 if variational else 1)))
        self.decoder = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Linear(128, 384))

    def encode(self, x):
        output = self.encoder(x)
        if self.variational:
            mean, logvar = output.chunk(2, -1)
            return mean, logvar.clamp(-8, 5)
        return output, None

    def reconstruction_loss(self, x):
        z, logvar = self.encode(x)
        kl = x.new_zeros(())
        if logvar is not None:
            kl = .5 * (z.square() + logvar.exp() - 1 - logvar).mean()
            z = z + (.5 * logvar).exp() * torch.randn_like(z)
        return F.mse_loss(self.decoder(z), x) + .001 * kl


def vicreg_loss(a, b):
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError("Paired batches with >=2 samples required")
    invariance = F.mse_loss(a, b)
    variance = sum(F.relu(1 - torch.sqrt(x.var(0) + 1e-4)).mean() for x in (a, b)) / 2
    covariance = a.new_zeros(())
    mask = ~torch.eye(a.shape[1], dtype=torch.bool, device=a.device)
    for x in (a, b):
        centered = x - x.mean(0)
        cov = centered.T @ centered / (len(x) - 1)
        covariance = covariance + cov[mask].square().sum() / a.shape[1]
    return 25 * invariance + 25 * variance + covariance


def validate_split(split, rows):
    arrays = {key: np.asarray(split[key], dtype=np.int64)
              for key in ("train", "selection", "internal_audit")}
    combined = np.concatenate(list(arrays.values()))
    if (min(map(len, arrays.values())) == 0 or len(combined) != len(rows)
            or set(combined.tolist()) != set(range(len(rows)))):
        raise ValueError("Split must partition the entire development cache exactly once")
    sets = []
    for name, ids in arrays.items():
        expected = "train" if name == "train" else "validation"
        if any(rows[i]["split"] != expected for i in ids):
            raise ValueError("Unexpected split or test row")
        sets.append({str(rows[i]["group_id"]) for i in ids})
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Exact-distance groups cross splits")
    if all("distance_m" in row for row in rows):
        labels = {}
        for row in rows:
            labels.setdefault(str(row["group_id"]), set()).add(round(float(row["distance_m"]), 9))
        if any(len(values) != 1 for values in labels.values()):
            raise ValueError("Same-distance pairing group contains distinct physical states")
    return arrays


def error_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite scalar errors required")
    absolute = np.abs(values)
    return dict(count=len(values), bias=float(values.mean()), mae=float(absolute.mean()),
                rmse=float(np.sqrt(np.mean(values ** 2))), p95=float(np.quantile(absolute, .95)),
                p99=float(np.quantile(absolute, .99)), max=float(absolute.max()))


def train_stats(z, train):
    # No validation/audit values may enter standardization.
    return z[train].mean(0), z[train].std(0).clamp_min(.01)


def fit_network(model, inputs, target, train, selection, steps, batch_size, lr=.001):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    generator = np.random.default_rng(7)
    best, best_score, best_step = None, float("inf"), 0
    for step in range(1, steps + 1):
        ids = generator.choice(train, batch_size)
        loss = F.mse_loss(model(inputs[ids]), target[ids])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                score = float(F.mse_loss(model(inputs[selection]), target[selection]))
            if score < best_score:
                best, best_score, best_step = copy.deepcopy(model.state_dict()), score, step
    model.load_state_dict(best); model.eval()
    for par in model.parameters():
        par.requires_grad_(False)
    return dict(best_step=best_step, selection_mse=best_score)


def fit_linear_probe(z, y, train, ridge=.001):
    # Mean loss + ridge*||w||^2; intercept is not penalized.
    design = torch.cat((z[train], z.new_ones(len(train), 1)), -1).double()
    penalty = torch.eye(design.shape[1], device=z.device, dtype=torch.float64) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design / len(train) + penalty,
                                 design.T @ y[train].double() / len(train))
    return weights.to(z.dtype)


def linear_readout(z, weights):
    return torch.cat((z, z.new_ones(len(z), 1)), -1) @ weights


def native_diagnostics(zi, zq, groups):
    centered = zi - zi.mean(0)
    total = float(centered.square().mean())
    within_terms = []
    for group in np.unique(groups):
        current = zi[np.flatnonzero(groups == group)]
        within_terms.append((current - current.mean(0)).square().mean())
    within = float(torch.stack(within_terms).mean())
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / max(len(zi) - 1, 1)).clamp_min(0)
    effective_rank = float(eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-20))
    delta = float((zi - zq).square().mean())
    return dict(native_dimension=zi.shape[1], centered_rms=total ** .5,
                collapsed=total <= 1e-10,
                relative_image_q_rmse=(delta / total) ** .5 if total > 1e-10 else None,
                within_distance_variance_fraction=within / total if total > 1e-10 else None,
                effective_rank_participation=effective_rank,
                note="Scale-relative supplementary metrics; low rank is not by itself failure in a 1-D distance task")


def write_report(output_dir, result):
    lines = ["# 原生表示多维度准确度对比", "", "同一开发图像，不经过旧R2距离头或actor。每种表示使用独立新训练的相同宽度MLP测量头；表中距离误差单位为米。",
             "", "旧R2是历史训练检查点；其他表示使用本轮固定预算，不是架构单变量消融。数据不是最终独立测试，结果不代表闭环安全。",
             "", "|方案|维度|图像距离MAE|RMSE|P95|最大|q距离MAE|图像q读出差MAE|线性头MAE|相对latent差|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in result["results"].items():
        e, q = r["mlp_probe"]["image_error_m"], r["mlp_probe"]["proxy_error_m"]
        relative = r["native"]["relative_image_q_rmse"]
        lines.append("|%s|%d|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%.6f|%s|" % (
            name, r["dimension"], e["mae"], e["rmse"], e["p95"], e["max"], q["mae"],
            r["mlp_probe"]["image_proxy_gap_m"]["mae"], r["linear_probe"]["image_error_m"]["mae"],
            "collapsed" if relative is None else "%.6f" % relative))
    lines += ["", "MLP头只测量表示中能读出的距离，不替换部署latent。不同维度的首层参数量不同，已逐项记录；线性头用于检查结论是否依赖非线性测量头。",
              "", "q为每种固定表示新拟合的距离代理，不是未来世界模型。Transformer等时序方法需另比较同历史与同动作下的1步、3步、10步误差。",
              "", "VICReg为同距离不同外观DINO特征配对适配，配对使用距离标签；AE/VAE是特征重建适配，不是原图端到端论文复现。"]
    (output_dir / "原生表示准确度对照.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--split", type=Path, default=Path("results/static_latent_benchmark_20261018_qcontrol/split.json"))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--families", nargs="+", choices=["pca", "ae", "vae", "vicreg"], default=["pca", "ae", "vae", "vicreg"])
    parser.add_argument("--codec-steps", type=int, default=800)
    parser.add_argument("--q-steps", type=int, default=1000)
    parser.add_argument("--probe-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir.exists() or len(set(args.dimensions)) != len(args.dimensions)
            or not all(2 <= d <= 384 for d in args.dimensions)
            or min(args.codec_steps, args.q_steps, args.probe_steps, args.batch_size) < 2
            or len(set(args.families)) != len(args.families)):
        parser.error("New output, unique dimensions/families, and positive budgets required")
    torch.set_num_threads(1)
    started = time.monotonic()
    paths = [args.cache_dir / "cache_manifest.json", args.cache_dir / "features.h5", args.representation, args.split, Path(__file__)]
    sources = {str(p): sha256(p) for p in paths}
    manifest = json.loads(paths[0].read_text()); rows = manifest["records"]
    if manifest.get("test_encoded") != 0 or sources[str(paths[1])] != manifest["feature_sha256"]:
        raise ValueError("Cache fingerprint or test guard failed")
    split = validate_split(json.loads(args.split.read_text()), rows)
    train, selection, audit = (split[k] for k in ("train", "selection", "internal_audit"))
    with h5py.File(paths[1], "r") as handle:
        raw = handle["features"][:]
        if (not handle.attrs.get("complete", False)
                or handle["filename"].asstr()[:].tolist() != [r["filename"] for r in rows]
                or not np.allclose(handle["distance_m"][:], [r["distance_m"] for r in rows], atol=1e-9, rtol=0)):
            raise ValueError("Cache incomplete or metadata/order mismatch")
    if raw.shape != (len(rows), 384) or not np.isfinite(raw).all():
        raise ValueError("Invalid features")
    device = torch.device(args.device)
    raw = torch.as_tensor(raw, device=device)
    xm, xs = raw[train].mean(0), raw[train].std(0).clamp_min(.01)
    x = (raw - xm) / xs
    distance = torch.tensor([r["distance_m"] for r in rows], device=device, dtype=torch.float32)[:, None]
    dm, ds = distance[train].mean(), distance[train].std().clamp_min(.01)
    y = (distance - dm) / ds
    groups = np.array([str(r["group_id"]) for r in rows])
    train_groups = [np.flatnonzero((groups == g) & np.isin(np.arange(len(rows)), train))
                    for g in np.unique(groups[train])]
    args.output_dir.mkdir(parents=True)
    config = dict(dimensions=args.dimensions, families=args.families, codec_steps=args.codec_steps,
                  q_steps=args.q_steps, probe_steps=args.probe_steps, seed=7,
                  batch_size=args.batch_size, device=args.device, source_sha256=sources,
                  train_rows=len(train), selection_rows=len(selection), audit_rows=len(audit),
                  evaluator="own new linear/MLP probes on native coordinates; no R2 decoder or actor",
                  probe_hidden_width=32, normalization="train only", test_evaluated=False,
                  audit_scope="historical validation reuse, not an independent final test",
                  historical_R2_budget_matched=False, world_model_training=False, ppo_sbc_training=False)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "split.json").write_text(json.dumps({k: v.tolist() for k, v in split.items()}))
    rep = torch.load(args.representation, map_location=device)
    r2 = SafetyProjection().to(device)
    r2.load_state_dict(rep["projection_state_dict"]); r2.eval().requires_grad_(False)
    with torch.no_grad():
        r2z = r2((raw - rep["feature_mean"].to(device)) / rep["feature_std"].to(device))
        xc = x[train] - x[train].mean(0)
        _, eigenvectors = torch.linalg.eigh(xc.T @ xc / (len(train) - 1))
        pca_mean = x[train].mean(0)
    arms = [("historical_R2", "r2", 32), ("DINO_CLS", "raw", 384)]
    arms += [(family + str(d), family, d) for family in args.families for d in args.dimensions]
    results = {}
    for name, family, dimension in arms:
        arm_start = time.monotonic(); torch.manual_seed(7)
        codec = None; best_state = None; best_step = 0
        if family == "r2":
            z = r2z
        elif family == "raw":
            z = x
        elif family == "pca":
            basis = eigenvectors[:, -dimension:].flip(-1)
            z = (x - pca_mean) @ basis
        else:
            codec = FeatureCodec(dimension, variational=(family == "vae")).to(device)
            optimizer = torch.optim.AdamW(codec.parameters(), lr=.001, weight_decay=1e-4)
            rng = np.random.default_rng(7); best_score = float("inf")
            for step in range(1, args.codec_steps + 1):
                if family == "vicreg":
                    # Same physical distance, different observed appearance, no synthetic RGB views.
                    chosen = rng.integers(len(train_groups), size=args.batch_size)
                    paired = [rng.choice(train_groups[k], 2, replace=len(train_groups[k]) < 2) for k in chosen]
                    left, right = np.asarray(paired).T
                    loss = vicreg_loss(codec.encode(x[left])[0], codec.encode(x[right])[0])
                else:
                    ids = rng.choice(train, args.batch_size)
                    loss = codec.reconstruction_loss(x[ids])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite codec loss")
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(codec.parameters(), 5.); optimizer.step()
                if step == 1 or step % 100 == 0 or step == args.codec_steps:
                    with torch.no_grad():
                        zz, _ = codec.encode(x[selection])
                        if family == "vicreg":
                            # Fixed nonoverlapping same-distance selection pairs, never audit labels.
                            pairs = [np.flatnonzero(groups[selection] == g)[:2] for g in np.unique(groups[selection])]
                            pair = np.stack(pairs)
                            score = float(vicreg_loss(zz[pair[:, 0]], zz[pair[:, 1]]))
                        else:
                            score = float(F.mse_loss(codec.decoder(zz), x[selection]))
                    if score < best_score:
                        best_score, best_state, best_step = score, copy.deepcopy(codec.state_dict()), step
            codec.load_state_dict(best_state); codec.eval().requires_grad_(False)
            with torch.no_grad():
                z = codec.encode(x)[0]
        z = z.detach()
        lm, ls = train_stats(z, train)
        standardized = (z - lm) / ls
        torch.manual_seed(7)
        q = PhysicalToLatent(latent_dim=dimension).to(device)
        q_training = fit_network(q, y, standardized, train, selection, args.q_steps, args.batch_size)
        with torch.no_grad():
            zq_standardized = q(y[audit])
            zq = zq_standardized * ls + lm
        # Freeze representation, then fit identical-width fresh measurement head.
        torch.manual_seed(7)
        probe = nn.Sequential(nn.Linear(dimension, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
        probe_training = fit_network(probe, standardized, y, train, selection, args.probe_steps, args.batch_size)
        ridge = fit_linear_probe(standardized, y, train)
        per_pair = {}
        def probe_results(image, proxy, prefix):
            image_d = (image * ds + dm).flatten()
            proxy_d = (proxy * ds + dm).flatten()
            image_error = (image_d - distance[audit].flatten()).cpu().numpy()
            proxy_error = (proxy_d - distance[audit].flatten()).cpu().numpy()
            gap = (image_d - proxy_d).cpu().numpy()
            per_pair[prefix + "_image_error_m"] = image_error
            per_pair[prefix + "_proxy_error_m"] = proxy_error
            per_pair[prefix + "_gap_m"] = gap
            return dict(image_error_m=error_summary(image_error), proxy_error_m=error_summary(proxy_error),
                        image_proxy_gap_m=error_summary(gap))
        with torch.no_grad():
            mlp_result = probe_results(probe(standardized[audit]), probe(zq_standardized), "mlp")
            linear_result = probe_results(linear_readout(standardized[audit], ridge),
                                          linear_readout(zq_standardized, ridge), "linear")
            native = native_diagnostics(z[audit], zq, groups[audit])
        appearances = np.array([str(rows[i]["weather"]) + "|" + str(rows[i]["color"]) for i in audit])
        slices = {g: error_summary(per_pair["mlp_image_error_m"][appearances == g]) for g in np.unique(appearances)}
        result = dict(dimension=dimension, family=family, native=native, mlp_probe=mlp_result,
                      linear_probe=linear_result, appearance_image_errors_m=slices,
                      q_training=q_training, probe_training=probe_training, best_codec_step=best_step,
                      probe_parameters=sum(p.numel() for p in probe.parameters()),
                      encoder_parameters=sum(p.numel() for p in codec.encoder.parameters()) if codec else None,
                      runtime_seconds=time.monotonic() - arm_start)
        results[name] = result
        np.savez_compressed(args.output_dir / (name + "_audit.npz"), rows=audit, groups=groups[audit], **per_pair)
        torch.save(dict(codec_state_dict=best_state, q_state_dict=q.state_dict(), probe_state_dict=probe.state_dict(),
                        linear_weights=ridge, latent_mean=lm, latent_std=ls, distance_mean=dm, distance_std=ds,
                        feature_mean=xm, feature_std=xs,
                        pca_mean=pca_mean if family == "pca" else None,
                        pca_basis=basis if family == "pca" else None, config=config, result=result),
                   args.output_dir / (name + ".pt"))
        partial = dict(status="running", config=config, results=results)
        (args.output_dir / "metrics.json").write_text(json.dumps(partial, indent=2, allow_nan=False))
        print("%s dim=%d image_MAE=%.6f proxy_MAE=%.6f gap_MAE=%.6f" % (
            name, dimension, mlp_result["image_error_m"]["mae"], mlp_result["proxy_error_m"]["mae"],
            mlp_result["image_proxy_gap_m"]["mae"]), flush=True)
    for path, digest in sources.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Frozen source changed")
    output = dict(status="native_representation_development_audit_complete", config=config, results=results,
                  source_files_unchanged=True, test_evaluated=False, runtime_seconds=time.monotonic() - started)
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2, allow_nan=False))
    write_report(args.output_dir, output)
    print("metrics:", args.output_dir / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
