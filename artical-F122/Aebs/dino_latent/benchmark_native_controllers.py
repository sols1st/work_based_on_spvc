"""Fresh native PPO per representation, then physical speed-residual audits.

No distance readout in the policy. Each input is [own native standardized z, v].
Reference trajectories are reused/off-policy diagnostics, NOT fresh deployment
CP. Library rollouts are not online CARLA. Old models/backends are never edited.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import gymnasium as gym
from gymnasium import spaces
import h5py
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from Aebs.system.env import AebsEnv
from Aebs.system.outcomes import classify_terminal_outcome
from Aebs.conformal.calibrate_controller import conformal_summary, classify_batch
from Aebs.dino_latent.models import SafetyProjection, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.benchmark_native_representations import FeatureCodec, validate_split
from Aebs.dino_latent.calibrate_transition_contract import next_state_residual
from Aebs.dino_latent.error_tail_metrics import tail_summary, trajectory_tail_summary


THRESHOLDS = [.05, .1, .15, .1506854683160781]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


@torch.no_grad()
def encode_native(raw, checkpoint, representation):
    """Decode the saved coordinates exactly; never use the distance probe."""
    raw = torch.as_tensor(np.array(raw, copy=True), dtype=torch.float32)
    family = checkpoint["result"]["family"]
    dimension = int(checkpoint["result"]["dimension"])
    if family == "r2":
        projection = SafetyProjection().eval()
        projection.load_state_dict(representation["projection_state_dict"])
        native = projection((raw - representation["feature_mean"]) / representation["feature_std"])
    else:
        x = (raw - checkpoint["feature_mean"]) / checkpoint["feature_std"]
        if family == "raw":
            native = x
        elif family == "pca":
            native = (x - checkpoint["pca_mean"]) @ checkpoint["pca_basis"]
        elif family in {"ae", "vae", "vicreg"}:
            codec = FeatureCodec(dimension, family == "vae").eval()
            codec.load_state_dict(checkpoint["codec_state_dict"])
            native = codec.encode(x)[0]
        else:
            raise ValueError("Unsupported native family")
    z = ((native - checkpoint["latent_mean"]) / checkpoint["latent_std"]).numpy()
    if z.shape != (len(raw), dimension) or not np.isfinite(z).all():
        raise ValueError("Invalid native observation")
    return z.astype(np.float32)


class NativeInterface:
    def __init__(self, checkpoint, latents, distances, rows, train):
        self.dimension = int(checkpoint["result"]["dimension"])
        self.q = PhysicalToLatent(latent_dim=self.dimension).eval()
        self.q.load_state_dict(checkpoint["q_state_dict"])
        self.q.requires_grad_(False)
        self.dm = float(checkpoint["distance_mean"])
        self.ds = float(checkpoint["distance_std"])
        self.latents = np.asarray(latents, dtype=np.float32)
        self.distances = np.asarray(distances, dtype=np.float32)
        self.pools = {}
        for index in train:
            key = str(rows[index]["weather"]) + "|" + str(rows[index]["color"])
            self.pools.setdefault(key, []).append(int(index))
        self.pools = {key: np.asarray(ids) for key, ids in sorted(self.pools.items())}
        # np.unique's first index retains original-library tie behavior.
        self.lookups = {}
        for key, pool in self.pools.items():
            values, first = np.unique(self.distances[pool], return_index=True)
            self.lookups[key] = (values, pool[first])

    @torch.no_grad()
    def proxy(self, distances):
        distances = np.asarray(distances, dtype=np.float32).reshape(-1)
        x = torch.from_numpy((distances[:, None] - self.dm) / self.ds)
        return self.q(x).numpy()

    def nearest(self, distances, appearance):
        values, indices = self.lookups[appearance]
        distances = np.asarray(distances, dtype=np.float64).reshape(-1)
        right = np.clip(np.searchsorted(values, distances), 0, len(values) - 1)
        left = np.clip(right - 1, 0, len(values) - 1)
        selected = np.where(abs(values[right] - distances) < abs(values[left] - distances), right, left)
        # If two different distances are equally close, original argmin picks
        # the earlier source row, not necessarily the smaller physical label.
        ties = abs(values[right] - distances) == abs(values[left] - distances)
        selected[ties] = np.where(indices[right[ties]] < indices[left[ties]], right[ties], left[ties])
        source = indices[selected]
        return self.latents[source], abs(values[selected] - distances), source


class NativePPOEnv(gym.Env):
    """Original physical/reward backend, new native observation only."""
    def __init__(self, interface, physical_scale, mode="mixed_episode"):
        super().__init__()
        self.interface, self.physical_scale, self.mode = interface, physical_scale, mode
        self.base = AebsEnv(physical_scale, max_episode_steps=400)
        self.action_space = self.base.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(interface.dimension + 1,), dtype=np.float32)
        self.active_mode = "surrogate"
        self.appearance = next(iter(interface.pools))

    def observe(self, state):
        d = float(state[0]) * self.physical_scale
        if self.active_mode == "surrogate":
            z = self.interface.proxy([d])[0]
        else:
            z = self.interface.nearest([d], self.appearance)[0][0]
        return np.append(z, np.float32(state[1])).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.active_mode = ("surrogate" if np.random.random() < .5 else "image") if self.mode == "mixed_episode" else self.mode
        self.appearance = list(self.interface.pools)[np.random.randint(len(self.interface.pools))]
        state, info = self.base.reset(seed=seed, options=options)
        return self.observe(state), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.base.step(action)
        return self.observe(state), reward, terminated, truncated, info


class LogProgress(BaseCallback):
    def __init__(self, path):
        super().__init__(); self.path = path; self.last = 0; self.started = time.monotonic()

    def _on_step(self):
        if self.num_timesteps - self.last >= 10000:
            self.last = self.num_timesteps
            save_json(self.path, dict(timesteps=self.num_timesteps, seconds=time.monotonic() - self.started))
        return True


def batch_step(states, actions, scale):
    states = np.asarray(states, dtype=np.float32)
    d = states[:, 0].astype(np.float64) * scale
    v = states[:, 1].astype(np.float64)
    a = np.clip(np.asarray(actions, dtype=np.float64).reshape(-1), -3, 3)
    nd, nv = d - .05 * v, np.clip(v - .05 * a, 0, 3)
    outcome = classify_batch(nd, nv, np.ones(len(d), dtype=bool), 1, 400)
    reward = 2 * (d - nd) - .001
    reward[outcome == "success"] += 2
    unsafe = outcome == "unsafe"
    reward[unsafe] -= (nv[unsafe] - .5) * 3
    return np.column_stack((nd / scale, nv)).astype(np.float32), reward, outcome


@torch.no_grad()
def actor_actions(model, z, speed):
    obs = np.column_stack((z, np.asarray(speed, dtype=np.float32))).astype(np.float32)
    return np.concatenate([model.predict(obs[i:i+8192], deterministic=True)[0].reshape(-1)
                           for i in range(0, len(obs), 8192)]).clip(-3, 3)


def library_rollouts(model, interface, physical_scale, grid_size, mode):
    d, v = np.meshgrid(np.linspace(15, 16, grid_size, dtype=np.float32),
                       np.linspace(2.5, 3, grid_size, dtype=np.float32), indexing="ij")
    init_d, init_v = d.reshape(-1), v.reshape(-1)
    labels = ["surrogate"] if mode == "surrogate" else list(interface.pools)
    n = len(init_d)
    states = np.column_stack((np.tile(init_d, len(labels)) / physical_scale,
                              np.tile(init_v, len(labels)))).astype(np.float32)
    active = np.ones(len(states), dtype=bool); outcomes = np.full(len(states), "", dtype=object)
    returns = np.zeros(len(states)); steps = np.zeros(len(states), dtype=np.int64)
    lookup_errors = []
    for step in range(1, 401):
        ids = np.flatnonzero(active)
        if not len(ids): break
        ds = states[ids, 0].astype(np.float64) * physical_scale
        if mode == "surrogate":
            z = interface.proxy(ds)
        else:
            z = np.empty((len(ids), interface.dimension), np.float32)
            for j, appearance in enumerate(labels):
                keep = ids // n == j
                if keep.any():
                    z[keep], errors, _ = interface.nearest(ds[keep], appearance)
                    lookup_errors.extend(errors.tolist())
        action = actor_actions(model, z, states[ids, 1])
        new_states, reward, events = batch_step(states[ids], action, physical_scale)
        if step == 400: events[events == ""] = "timeout"
        states[ids] = new_states; returns[ids] += reward; steps[ids] = step
        finished = events != ""
        outcomes[ids[finished]] = events[finished]; active[ids[finished]] = False
    def stats(mask):
        count = Counter(outcomes[mask].tolist())
        return dict(episodes=int(mask.sum()), counts=dict(count),
                    **{key + "_rate": count[key] / int(mask.sum()) for key in (
                        "success", "unsafe", "timeout", "stopped_safe_outside_goal", "out_of_domain")},
                    mean_return=float(returns[mask].mean()), mean_steps=float(steps[mask].mean()))
    result = stats(np.ones(len(states), dtype=bool))
    result["by_appearance"] = {name: stats(np.arange(len(states)) // n == j) for j, name in enumerate(labels)}
    result["lookup_error_m"] = dict(mean=float(np.mean(lookup_errors)), max=float(np.max(lookup_errors))) if lookup_errors else None
    result["scope"] = "analytic physical dynamics; train-library nearest RGB feature replay, NOT online CARLA" if mode != "surrogate" else "own q proxy in analytic physical environment"
    return result


def matched_speed_audit(model, interface, distances, indices, output):
    residuals, image_actions, q_actions, image_rows, speeds = [], [], [], [], []
    grid = np.linspace(0, 3, 31, dtype=np.float32)
    for start in range(0, len(indices), 256):
        ii = np.repeat(indices[start:start+256], len(grid))
        vv = np.tile(grid, len(indices[start:start+256])); dd = distances[ii]
        keep = np.array([classify_terminal_outcome(a, b) is None for a, b in zip(dd, vv)])
        ii, vv, dd = ii[keep], vv[keep], dd[keep]
        if not len(ii): continue
        ai = actor_actions(model, interface.latents[ii], vv)
        aq = actor_actions(model, interface.proxy(dd), vv)
        residual = next_state_residual(dd, vv, ai, aq)[:, 1]
        residuals.append(residual); image_actions.append(ai); q_actions.append(aq); image_rows.append(ii); speeds.append(vv)
    residual = np.concatenate(residuals)
    ai, aq, ii, vv = map(np.concatenate, (image_actions, q_actions, image_rows, speeds))
    ni = np.clip(vv - .05 * ai, 0, 3); nq = np.clip(vv - .05 * aq, 0, 3)
    nd = distances[ii] - .05 * vv
    unsafe_i, unsafe_q = (nd <= 6) & (ni > .5), (nd <= 6) & (nq > .5)
    np.savez_compressed(output, residual_mps=residual, image_action=ai, q_action=aq,
                        feature_row=ii, speed_mps=vv, distance_m=distances[ii])
    return dict(images=len(indices), steps=tail_summary(residual, THRESHOLDS),
                image_only_next_unsafe=int((unsafe_i & ~unsafe_q).sum()),
                q_only_next_unsafe=int((unsafe_q & ~unsafe_i).sum()), both_next_unsafe=int((unsafe_i & unsafe_q).sum()),
                scope="exact same RGB-label state and speed; no nearest-image lookup")


def reference_residual_audit(model, interface, checkpoint, representation, panel, output):
    raw = panel["features"]; d, v, group = panel["distance_m"], panel["speed_mps"], panel["trajectory_index"]
    z = encode_native(raw, checkpoint, representation)
    ai = actor_actions(model, z, v); aq = actor_actions(model, interface.proxy(d), v)
    residual = next_state_residual(d, v, ai, aq)[:, 1]
    ts, scores = trajectory_tail_summary(residual, group, int(panel["trajectory_count"]), THRESHOLDS)
    cp = conformal_summary(scores, .01, .01)
    worst = int(np.argmax(abs(residual)))
    np.savez_compressed(output, residual_mps=residual, trajectory_index=group, trajectory_score_mps=scores,
                        image_action=ai, q_action=aq, distance_m=d, speed_mps=v)
    return dict(step=tail_summary(residual, THRESHOLDS), trajectory_score=ts,
                same_recipe_radius_mps=cp["q_hat"], rank_from_largest=cp["rank_from_largest_l"],
                nominal_iid_confidence=cp["achieved_confidence"], target_trajectory_coverage=.99,
                fresh_CP_guarantee=False, safety_certificate=False,
                scope="reused 459 old PPO09 reference trajectories; off-policy common-state diagnostic, NOT candidate on-policy calibration",
                worst=dict(distance_m=float(d[worst]), speed_mps=float(v[worst]),
                           image_action=float(ai[worst]), q_action=float(aq[worst]), residual_mps=float(residual[worst])))


def run_one(config, name):
    torch.set_num_threads(1)
    output = Path(config["output_dir"]) / name
    output.mkdir()
    save_json(output / "status.json", dict(status="loading", name=name))
    try:
        started = time.monotonic()
        ck = torch.load(Path(config["native_dir"]) / (name + ".pt"), map_location="cpu")
        rep = torch.load(config["representation"], map_location="cpu")
        manifest = json.loads((Path(config["cache_dir"]) / "cache_manifest.json").read_text())
        rows = manifest["records"]
        split = validate_split(json.loads((Path(config["native_dir"]) / "split.json").read_text()), rows)
        with h5py.File(Path(config["cache_dir"]) / "features.h5", "r") as f:
            raw, distances = f["features"][:], f["distance_m"][:].astype(np.float32)
        latent = encode_native(raw, ck, rep)
        interface = NativeInterface(ck, latent, distances, rows, split["train"])
        env = NativePPOEnv(interface, float(rep["distance_scale_m"]))
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, n_epochs=10, gamma=.99, gae_lambda=.95,
                    ent_coef=.01, seed=7, device="cpu", policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])))
        save_json(output / "status.json", dict(status="training", name=name, requested_timesteps=config["timesteps"]))
        train_start = time.monotonic()
        model.learn(total_timesteps=config["timesteps"], callback=LogProgress(output / "progress.json"))
        model.save(output / "native_ppo.zip")
        result = dict(name=name, native_dimension=interface.dimension, seed=7,
                      requested_timesteps=config["timesteps"], actual_timesteps=int(model.num_timesteps),
                      training_seconds=time.monotonic() - train_start,
                      controller_sha256=sha256(output / "native_ppo.zip"),
                      representation_sha256=sha256(Path(config["native_dir"]) / (name + ".pt")),
                      observation="own native standardized latent + measured physical speed; no distance readout",
                      actor_parameters=sum(p.numel() for p in model.policy.mlp_extractor.policy_net.parameters()) + sum(p.numel() for p in model.policy.action_net.parameters()))
        save_json(output / "status.json", dict(status="evaluating", name=name))
        # Audit only after saving the predetermined final PPO, never select on residual/CP.
        panel = np.load(Path(config["output_dir"]) / "reference_panel.npz")
        result["reference_residual"] = reference_residual_audit(model, interface, ck, rep, panel, output / "reference_residual.npz")
        result["matched"] = {}
        for key in ("train", "internal_audit"):
            result["matched"][key] = matched_speed_audit(model, interface, distances, split[key], output / (key + "_matched.npz"))
        result["rollouts"] = {mode: library_rollouts(model, interface, float(rep["distance_scale_m"]), config["eval_grid_size"], mode)
                              for mode in ("surrogate", "image")}
        result["sampled_task_gate"] = all(r["success_rate"] >= .95 and r["unsafe_rate"] <= .01 and r["timeout_rate"] <= .01
                                          for r in result["rollouts"].values())
        result["task_gate_scope"] = "development performance screen only; does not constitute a safety proof"
        result.update(status="native_ppo_speed_residual_development_complete", runtime_seconds=time.monotonic() - started,
                      SBC_QP_CP_training=False, RGB_test_used=False)
        save_json(output / "metrics.json", result)
        save_json(output / "status.json", dict(status="complete", name=name))
        return name, result
    except Exception:
        save_json(output / "status.json", dict(status="failed", name=name, traceback=traceback.format_exc()))
        raise


def report(output, result):
    lines = ["# 各原生表示新控制器的速度残差对比", "", "每个方案从头训练自己的PPO。误差单位为米/秒，计算同一当前物理状态下图像动作与q动作造成的下一速度差。距离测量头不进入控制器。",
             "", "459条旧PPO09轨迹已被多轮开发复用，本表半径沿用旧CP统计配方，但不是新的独立CP保证。控制任务表现另用训练图片库闭环检查，不等于在线CARLA。",
             "", "|方案|维度|参考残差MAE|P95|最大及同配方半径|超过0.15步数|代理成功率|图片库成功率|图片库unsafe率|任务门槛|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name in result["config"]["arms"]:
        if name not in result["results"]: continue
        r = result["results"][name]; e = r["reference_residual"]["step"]
        lines.append("|%s|%d|%.6f|%.6f|%.6f|%d/%d|%.2f%%|%.2f%%|%.2f%%|%s|" % (
            name, r["native_dimension"], e["mae"], e["p95_abs"], r["reference_residual"]["same_recipe_radius_mps"],
            e["thresholds"]["0.15"]["count_above"], e["count"],
            100 * r["rollouts"]["surrogate"]["success_rate"], 100 * r["rollouts"]["image"]["success_rate"],
            100 * r["rollouts"]["image"]["unsafe_rate"], "通过" if r["sampled_task_gate"] else "未通过"))
    lines += ["", "本表只比较新训练的原生接口；不是已有最小SPVC最终actor的替换证书。不同输入维度导致第一层参数量不同，预算、隐藏宽度、训练混合比例及评价状态固定。",
              "", "0.15来自既往问题的数值参照，不是通用安全阈值。半径小但任务门槛未通过的方案，不能当作更好控制器。完整P99、正负残差、精确图像单步数据和各天气表现见metrics.json及各子目录。"]
    (output / "原生控制器速度残差对照.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("results/dino_expanded_v1/02_dino_h5"))
    parser.add_argument("--representation", type=Path, default=Path("results/dino_expanded_v1/08_R2_group_alignment/dino_safety_latent.pt"))
    parser.add_argument("--reference-h5", type=Path, default=Path("results/dino_expanded_v1/20_trajectory_sbc_cp_calibration_v1/trajectories.h5"))
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--eval-grid-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or min(args.timesteps, args.workers) < 1 or args.eval_grid_size < 2:
        parser.error("New output, positive budgets and grid>=2 required")
    torch.set_num_threads(1)
    source = json.loads((args.native_dir / "metrics.json").read_text())
    if source["status"] != "native_representation_development_audit_complete" or source["test_evaluated"]:
        raise ValueError("Incomplete representation benchmark or test used")
    for path, digest in source["config"]["source_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError("Native representation source changed: " + path)
    arms = args.arms or list(source["results"])
    if len(set(arms)) != len(arms) or not set(arms) <= set(source["results"]):
        raise ValueError("Invalid arms")
    files = [args.native_dir / "metrics.json", args.native_dir / "split.json", args.representation,
             args.cache_dir / "cache_manifest.json", args.cache_dir / "features.h5", args.reference_h5, Path(__file__)]
    files += [args.native_dir / (name + ".pt") for name in arms]
    hashes = {str(p): sha256(p) for p in files}
    manifest = json.loads((args.cache_dir / "cache_manifest.json").read_text())
    if manifest.get("test_encoded") != 0 or manifest["feature_sha256"] != hashes[str(args.cache_dir / "features.h5")]:
        raise ValueError("Cache hash or no-test guard failed")
    validate_split(json.loads((args.native_dir / "split.json").read_text()), manifest["records"])
    args.output_dir.mkdir()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(arms=arms, seed=7, source_sha256=hashes, no_distance_probe=True,
                  training="fresh PPO, 50% own q and 50% train-image episodes, uniform fixed appearance per episode",
                  same_PPO_hyperparameters=True, checkpoint_selection="fixed final timestep only; no reference-residual selection",
                  reference_scope="459 reused PPO09 reference trajectories, not own-policy/fresh CP calibration",
                  task_gate="surrogate and train-library success>=95%, unsafe<=1%, timeout<=1%; sampled development only")
    save_json(args.output_dir / "config.json", config)
    with h5py.File(args.reference_h5, "r") as f:
        mask = f["valid"][:].astype(bool); ep, tt = np.where(mask)
        ids = f["trajectory_id"].asstr()[:]
        if len(mask) != 459 or len(set(ids)) != 459 or not f.attrs.get("complete"):
            raise ValueError("Wrong reference panel")
        raw = f["dino_features"][:][ep, tt]
        d, v = f["distance_m"][:][ep, tt], f["speed_mps"][:][ep, tt]
        reference_provenance = json.loads(f.attrs["provenance_json"])
        if reference_provenance.get("representation_sha256") != sha256(args.representation):
            raise ValueError("Reference features were collected with a different representation")
    config["reference_collection_provenance"] = reference_provenance
    config["reference_steps"] = len(d)
    save_json(args.output_dir / "config.json", config)
    if not all(np.isfinite(x).all() for x in (raw, d, v)):
        raise ValueError("Nonfinite reference data")
    np.savez_compressed(args.output_dir / "reference_panel.npz", features=raw, distance_m=d, speed_mps=v,
                        trajectory_index=ep, time_index=tt, trajectory_count=len(mask))
    result = dict(status="running", config=config, results={}, failures={})
    save_json(args.output_dir / "metrics.json", result)
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, config, name): name for name in arms}
        for future in as_completed(futures):
            name = futures[future]
            try:
                _, metrics = future.result(); result["results"][name] = metrics
                e = metrics["reference_residual"]["step"]
                print("%s COMPLETE residual_MAE=%.6f radius=%.6f surrogate_success=%.2f%% image_success=%.2f%% gate=%s" % (
                    name, e["mae"], metrics["reference_residual"]["same_recipe_radius_mps"],
                    100 * metrics["rollouts"]["surrogate"]["success_rate"],
                    100 * metrics["rollouts"]["image"]["success_rate"], metrics["sampled_task_gate"]), flush=True)
            except Exception:
                result["failures"][name] = traceback.format_exc()
                print(name + " FAILED; inspect status.json", flush=True)
            save_json(args.output_dir / "metrics.json", result); report(args.output_dir, result)
    changed = {str(path): sha256(Path(path)) != digest for path, digest in hashes.items()}
    if any(changed.values()): raise ValueError("Frozen source changed: %s" % changed)
    result.update(status="native_controller_development_comparison_complete" if not result["failures"] else "partial_with_failures",
                  source_files_unchanged=True, runtime_seconds=time.monotonic() - started)
    save_json(args.output_dir / "metrics.json", result); report(args.output_dir, result)
    print("metrics:", args.output_dir / "metrics.json", flush=True)
    if result["failures"]: raise RuntimeError("Some candidates failed; completed artifacts preserved")


if __name__ == "__main__": main()
