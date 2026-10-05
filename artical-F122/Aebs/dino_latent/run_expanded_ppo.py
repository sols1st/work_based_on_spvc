"""Fresh PPO on frozen R1, train-only appearance replay, matched diagnostics."""
import argparse
import heapq
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.models import SafetyProjection
from Aebs.dino_latent.train_latent_ppo import LatentAebsEnv, evaluate
from Aebs.dino_latent.holdout_experiment import Progress
from Aebs.dino_latent.diagnose_matched_images import paired_observations, one_step
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.system.outcomes import classify_terminal_outcome


class AppearanceEnv(LatentAebsEnv):
    """Keep one weather/color per episode; only train rows are provided."""
    def __init__(self, checkpoint, data, latent, rows, mode, appearance=None):
        ids = [i for i,r in enumerate(rows) if r["split"] == "train"]
        super().__init__(checkpoint, mode, data_path=data, latent_data_path=latent, image_indices=ids)
        self.pools = {}
        for local, i in enumerate(ids):
            key = rows[i]["weather"]+"|"+rows[i]["color"]
            self.pools.setdefault(key, []).append(local)
        self.pools = {k:np.asarray(v) for k,v in self.pools.items()}
        self.fixed_appearance = appearance
        self.appearance = appearance or sorted(self.pools)[0]

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if self.fixed_appearance is None:
            self.appearance = sorted(self.pools)[np.random.randint(len(self.pools))]
        return self._observation(self.base.state), info

    def _latent(self, distance_norm):
        if self.active_observation_mode == "surrogate":
            return super()._latent(distance_norm)
        pool = self.pools[self.appearance]
        d = distance_norm*self.distance_scale_m
        i = pool[np.argmin(np.abs(self.image_distances_m[pool]-d))]
        self.lookup_errors_m.append(float(abs(self.image_distances_m[i]-d)))
        return self.image_latents[i].astype(np.float32)


def matched(model, env, latents, distances, ids, rows):
    counts, excluded = Counter(), Counter()
    total, absolute, maximum, less = 0, 0., 0., 0
    worst = []
    for start in range(0,len(ids),64):
        image_ids,d,v,oi,oq = paired_observations(latents,distances,ids[start:start+64],
            np.linspace(0,3,31,dtype=np.float32),env.q_model,env.distance_scale_m)
        ai = model.predict(oi,deterministic=True)[0].reshape(-1)
        aq = model.predict(oq,deterministic=True)[0].reshape(-1)
        if not np.isfinite(ai).all() or not np.isfinite(aq).all():
            raise ValueError("Nonfinite actions")
        for j in range(len(d)):
            terminal = classify_terminal_outcome(float(d[j]),float(v[j]))
            if terminal:
                excluded[terminal] += 1
                continue
            a,b = float(np.clip(ai[j],-3,3)),float(np.clip(aq[j],-3,3))
            ni = one_step(env.base,float(d[j]),float(v[j]),a)
            nq = one_step(env.base,float(d[j]),float(v[j]),b)
            unsafe_i,unsafe_q = ni["outcome"] == "unsafe",nq["outcome"] == "unsafe"
            counts["image_only_next_unsafe"] += int(unsafe_i and not unsafe_q)
            counts["surrogate_only_next_unsafe"] += int(unsafe_q and not unsafe_i)
            counts["both_next_unsafe"] += int(unsafe_i and unsafe_q)
            counts["outcome_disagreement"] += int(ni["outcome"] != nq["outcome"])
            error = abs(a-b)
            total += 1; absolute += error; maximum = max(maximum,error); less += int(a < b-1e-6)
            detail = dict(filename=rows[int(image_ids[j])]["filename"],distance_m=float(d[j]),
                          speed=float(v[j]),image_action=a,surrogate_action=b,image_next=ni,surrogate_next=nq)
            heapq.heappush(worst,(error,total,detail))
            if len(worst)>10:
                heapq.heappop(worst)
        if start % 1024 == 0:
            print("matched images %d/%d" % (min(start+64,len(ids)),len(ids)),flush=True)
    return dict(images=len(ids),checked_nonterminal_pairs=total,excluded=dict(excluded),
        action_mae=absolute/total if total else None,action_max_abs=maximum if total else None,
        image_less_braking_fraction=less/total if total else None,lookup_distance_error_m=0,
        **counts,worst_samples=[r[2] for r in sorted(worst,reverse=True)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,default=Path("results/dino_expanded_v1/02_dino_h5"))
    p.add_argument("--representation",type=Path,default=Path("results/dino_expanded_v1/03_alignment_r0_r1/R1_all_appearances/dino_safety_latent.pt"))
    p.add_argument("--output-dir",type=Path,default=Path("results/dino_expanded_v1/04_R1_ppo"))
    p.add_argument("--timesteps",type=int,default=200000)
    p.add_argument("--eval-grid-size",type=int,default=20)
    args = p.parse_args()
    if args.output_dir.exists() or args.timesteps<1 or args.eval_grid_size<2:
        p.error("New output directory and valid sizes required")
    torch.set_num_threads(1)
    cache_path = args.cache_dir/"cache_manifest.json"
    cache = json.loads(cache_path.read_text())
    rep = torch.load(args.representation,map_location="cpu")
    if (cache.get("schema") != "spvc_expanded_dino_h5_cache_v1" or cache.get("test_encoded") != 0 or
            rep["provenance"]["cache_manifest_sha256"] != sha256(cache_path)):
        p.error("Representation/cache provenance mismatch")
    rows = cache["records"]
    if any(r["split"] not in ("train","validation") for r in rows):
        p.error("Test rows forbidden")
    if sha256(args.cache_dir/"features.h5") != cache["feature_sha256"]:
        p.error("H5 changed")
    projection = SafetyProjection().eval()
    projection.load_state_dict(rep["projection_state_dict"])
    with h5py.File(args.cache_dir/"features.h5","r") as stream:
        if not stream.attrs["complete"]:
            p.error("Incomplete H5")
        features = torch.from_numpy(stream["features"][:])
        distances = stream["distance_m"][:].astype(np.float32)
        if stream["filename"].asstr()[:].tolist() != [r["filename"] for r in rows]:
            p.error("H5 row mismatch")
    with torch.no_grad():
        latents = projection((features-rep["feature_mean"])/rep["feature_std"]).numpy()
    if not np.isfinite(latents).all():
        raise ValueError("Invalid projected latent")
    args.output_dir.mkdir(parents=True)
    config = dict(representation_sha256=sha256(args.representation),cache_manifest_sha256=sha256(cache_path),
        timesteps=args.timesteps,seed=7,test_used=False,eval_grid_size=args.eval_grid_size,
        replay="50% surrogate / 50% train images per episode; uniform fixed weather/color per episode",
        limitation="train-library nearest-distance replay is not online image rollout; validation matched one-step only")
    (args.output_dir/"config.json").write_text(json.dumps(config,indent=2))
    data = args.output_dir/"physical_labels.h5"
    latent = args.output_dir/"latent_data.npz"
    with h5py.File(data,"w") as stream:
        stream.create_dataset("y_train",data=distances)
        stream.attrs["note"] = "Adapter row labels include train/validation; train image_indices always enforced"
    np.savez_compressed(latent,latent=latents,available_indices=np.arange(len(rows)))
    def make_env(mode,appearance=None):
        return AppearanceEnv(args.representation,data,latent,rows,mode,appearance)
    model = PPO("MlpPolicy",make_env("mixed_episode"),verbose=0,learning_rate=3e-4,n_steps=2048,
                batch_size=64,n_epochs=10,gamma=.99,gae_lambda=.95,ent_coef=.01,seed=7,device="cpu")
    print("Training fresh PPO, frozen R1, train-only replay",flush=True)
    model.learn(total_timesteps=args.timesteps,callback=Progress())
    model.save(args.output_dir/"latent_ppo.zip")
    result = dict(config,actual_timesteps=int(model.num_timesteps),checkpoint_sha256=sha256(args.output_dir/"latent_ppo.zip"),
                  evaluation={},matched={})
    result["evaluation"]["surrogate"] = evaluate(model,make_env("surrogate"),args.eval_grid_size)
    for appearance in sorted(make_env("image_nearest").pools):
        print("Evaluating train-library "+appearance,flush=True)
        result["evaluation"][appearance] = evaluate(model,make_env("image_nearest",appearance),args.eval_grid_size)
        (args.output_dir/"rollout_metrics.json").write_text(json.dumps(result["evaluation"],indent=2))
    for split in ("train","validation"):
        print("Matched one-step "+split,flush=True)
        ids = [i for i,r in enumerate(rows) if r["split"] == split]
        result["matched"][split] = matched(model,make_env("surrogate"),latents,distances,ids,rows)
        (args.output_dir/"metrics.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    print("[Expanded R1 PPO and matched diagnostic — no SBC/test]")
    for name,m in result["evaluation"].items():
        print(name,{k:m[k] for k in ["success_rate","unsafe_rate","timeout_rate","mean_steps","nearest_image_distance_error_m"]})
    for name,m in result["matched"].items():
        print(name,{k:v for k,v in m.items() if k != "worst_samples"})
    print("metrics:",args.output_dir/"metrics.json")
    print("Sampled development evidence only, not a safety certificate.")


if __name__ == "__main__":
    main()
