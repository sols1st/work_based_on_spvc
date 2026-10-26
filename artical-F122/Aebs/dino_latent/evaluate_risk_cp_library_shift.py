"""Stress a FROZEN trajectory-risk threshold by changing the image library.

No recalibration or fitting. Validation images have a different distance
lookup pattern. They are reused development data, not a new held-out test.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.benchmark_native_controllers import NativeInterface, actor_actions, encode_native
from Aebs.dino_latent.benchmark_native_representations import validate_split
from Aebs.dino_latent.experiment_sbc_risk_cp import FrozenBarrier, SCORE_NAMES, run_rollouts, save_json, summary
from Aebs.dino_latent.prepare_expanded_data import sha256


class QuantizedProxyInterface:
    """Counterfactual: replace the selected image latent by q(image label).

    Preserves lookup mismatch but removes the image-vs-q latent difference.
    This is NOT a deployable image policy or a newly trained controller.
    """
    def __init__(self, base):
        self.base=base; self.pools=base.pools; self.dimension=base.dimension

    def proxy(self, distances): return self.base.proxy(distances)

    def nearest(self, distances, appearance):
        _,errors,source=self.base.nearest(distances,appearance)
        return self.base.proxy(self.base.distances[source]),errors,source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-dir", type=Path, default=Path("results/native_R2_sbc_risk_cp_20261020_v2"))
    p.add_argument("--monitor-dir", type=Path, default=Path("results/native_R2_risk_monitor_20261020_v1"))
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists(): p.error("New output directory required")
    torch.set_num_threads(1)
    metrics = json.loads((a.experiment_dir/"metrics.json").read_text())
    sources = metrics["config"]["source_sha256"]
    def find(suffix): return next(Path(k) for k in sources if k.endswith(suffix))
    rep_path, native_path = find("/dino_safety_latent.pt"), find("/historical_R2.pt")
    controller_path, cache_path = find("/native_ppo.zip"), find("/features.h5")
    manifest_path, split_path = find("/cache_manifest.json"), find("/split.json")
    monitor_path = a.monitor_dir/"risk_monitor.pt"
    paths = [rep_path, native_path, controller_path, cache_path, manifest_path, split_path, monitor_path,
             a.monitor_dir/"metrics.json", a.experiment_dir/"metrics.json"]
    hashes = {str(path):sha256(path) for path in paths}
    for path in paths[:6] + [monitor_path]:
        if hashes[str(path)] != sources[str(path)]: raise ValueError("Frozen source identity mismatch")
    rep, ck = torch.load(rep_path,map_location="cpu"),torch.load(native_path,map_location="cpu")
    rows = json.loads(manifest_path.read_text())["records"]
    split = validate_split(json.loads(split_path.read_text()),rows)
    validation_indices = np.concatenate((split["selection"],split["internal_audit"]))
    with h5py.File(cache_path,"r") as f:
        raw,distances = f["features"][:],f["distance_m"][:]
    interface = NativeInterface(ck,encode_native(raw,ck,rep),distances,rows,validation_indices)
    ppo = PPO.load(controller_path,device="cpu")
    ppo.policy.set_training_mode(False); ppo.policy.requires_grad_(False)
    monitor_metrics = json.loads((a.monitor_dir/"metrics.json").read_text())
    n = metrics["config"]["rollout_counts"]["evaluation"]
    seed = metrics["config"]["seeds"]["evaluation"]
    horizon = metrics["config"]["horizon"]
    # The main monitor wrapper takes legacy metrics only as a default;
    # explicit normalization avoids using any legacy interval bound here.
    barrier = FrozenBarrier(monitor_path,{},metrics["config"]["barrier_normalization"]["distance_scale"],
                            monitor_metrics["normalization"])
    a.output_dir.mkdir(parents=True)
    save_json(a.output_dir/"config.json",dict(source_sha256=hashes, seed=seed, episodes=n,
              threshold_frozen=True, calibration_changed=False, library_rows=len(validation_indices),
              scope="validation-library shift development stress; no new unseen test or real CARLA"))
    result,scores,_ = run_rollouts(ppo,interface,barrier,n,seed,horizon,a.output_dir/"shift_rollouts.npz")
    if result["initial_state_sha256"] != metrics["rollouts"]["evaluation"]["initial_state_sha256"]:
        raise ValueError("Paired initial states differ")
    comparison = {}
    for name in SCORE_NAMES:
        q = metrics["trajectory_score_calibration"][name]["tolerance"]["q_hat"]
        comparison[name] = dict(frozen_q_hat=q,shift_coverage=float((scores[name] <= q).mean()),
                                shifted_negative_score_fraction=float((scores[name] < 0).mean()))
    quantized_result,_,_=run_rollouts(ppo,QuantizedProxyInterface(interface),barrier,n,seed,horizon,
                                    a.output_dir/"shift_quantized_proxy_rollouts.npz")
    # Algebraic decomposition at the SAME states of the actual image replay.
    saved=np.load(a.output_dir/"shift_rollouts.npz")
    ii,tt=np.nonzero(saved["valid"])
    states=saved["states_m_mps"][ii,tt]
    sources_idx=np.empty(len(ii),np.int64)
    for j,label in enumerate(saved["appearance_labels"]):
        keep=saved["appearance_index"][ii]==j
        if keep.any(): sources_idx[keep]=interface.nearest(states[keep,0],str(label))[2]
    aq_lookup=actor_actions(ppo,interface.proxy(interface.distances[sources_idx]),states[:,1])
    ai=saved["image_action"][ii,tt]; aq=saved["proxy_action"][ii,tt]
    vi=np.clip(states[:,1].astype(float)-.05*ai,0,3)
    vql=np.clip(states[:,1].astype(float)-.05*aq_lookup,0,3)
    vq=np.clip(states[:,1].astype(float)-.05*aq,0,3)
    lookup_error=vql-vq; latent_error=vi-vql; total_error=vi-vq
    if not np.allclose(total_error,lookup_error+latent_error,atol=1e-12,rtol=0):
        raise ValueError("Error decomposition does not telescope")
    np.savez_compressed(a.output_dir/"shift_error_decomposition.npz",states=states,source_row=sources_idx,
                        selected_image_distance_m=interface.distances[sources_idx],
                        lookup_speed_error_mps=lookup_error,latent_speed_error_mps=latent_error,total_speed_error_mps=total_error)
    decomposition=dict(scope="counterfactual decomposition at identical image-replay current states; absolute components need not add",
                       lookup_abs_mps=summary(abs(lookup_error)),latent_abs_mps=summary(abs(latent_error)),
                       total_abs_mps=summary(abs(total_error)),
                       identity="r_total=(v_q(image_label)-v_q(true_d))+(v_image-v_q(image_label))")
    for path,digest in hashes.items():
        if sha256(Path(path)) != digest: raise ValueError("Frozen source changed")
    output = dict(status="library_shift_stress_complete", rollouts=result, score_threshold_comparison=comparison,
                  quantized_proxy_only_lookup=quantized_result,error_decomposition=decomposition,
                  original_train_library_outcomes=metrics["rollouts"]["evaluation"]["outcomes"],
                  matched_initial_states=True, recalibration=False, model_training=False,
                  guarantee_transferred_to_shift=False, source_sha256=hashes)
    save_json(a.output_dir/"metrics.json",output)
    print("[Frozen threshold library shift]",result["outcomes"],"lookup",result["lookup_error_m"],flush=True)
    print("score coverage:",comparison,flush=True)
    print("q(image label) counterfactual:",quantized_result["outcomes"],"decomposition:",decomposition,flush=True)
    print("metrics:",a.output_dir/"metrics.json",flush=True)


if __name__ == "__main__": main()
