"""Frozen same-image distance/physical-successor A/B. No training, test or rollouts.

Current DINO uses the final rechecked minimal actor by default; legacy uses the shipped
cGAN/state_net/base PPO. These are existing systems, NOT equal-data retraining.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch
from stable_baselines3 import PPO

from Combined_network.model import AebsEnd2EndNet
from cGAN.taxi_models_and_data import AebsMLPGenerator
from Aebs.dino_latent.models import SafetyProjection, SafetyDecoder, PhysicalToLatent
from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.proxy_comparison_math import legacy_pixels, next_speed, errors, paired_summary, transition_summary
from Aebs.dino_latent.spvc_saved_actor import load_spvc_pair
from Aebs.system.outcomes import classify_terminal_outcome


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_ppo(model, input_dim):
    if (model.observation_space.shape != (input_dim,) or model.action_space.shape != (1,)
            or not np.array_equal(model.action_space.low, [-3])
            or not np.array_equal(model.action_space.high, [3])
            or getattr(model.policy, "squash_output", False)):
        raise ValueError("unsupported PPO interface")
    model.policy.set_training_mode(False)


def predict_actions(net, representation, speeds):
    n = len(representation)
    obs = np.concatenate([np.repeat(representation, len(speeds), axis=0),
                          np.tile(speeds, n)[:, None]], axis=1).astype(np.float32)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(obs), 4096):
            value = net(torch.from_numpy(obs[start:start+4096])).numpy().reshape(-1)
            if not np.isfinite(value).all():
                raise ValueError("nonfinite controller action")
            chunks.append(np.clip(value, -3, 3))
    return np.concatenate(chunks).reshape(n, len(speeds))


def main():
    root = Path("results/dino_expanded_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=root/"02_dino_h5")
    parser.add_argument("--data-root", type=Path, help="Override recorded RGB root after moving data")
    parser.add_argument("--representation", type=Path, default=root/"08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--dino-controller", type=Path, default=root/"09_R2_ppo/latent_ppo.zip")
    parser.add_argument("--spvc-dir", type=Path, help="Default depends on --dino-stage")
    parser.add_argument("--dino-stage", choices=("minimal", "spvc22", "ppo09"), default="minimal")
    parser.add_argument("--final-recheck", type=Path,
                        default=Path("results/dino_spvc_minimal_v1_final_recheck/metrics.json"))
    parser.add_argument("--legacy-controller", type=Path, default=Path("Aebs/controller/best_model/best_model.zip"))
    parser.add_argument("--legacy-generator", type=Path, default=Path("Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth"))
    parser.add_argument("--legacy-state-net", type=Path, default=Path("Aebs/controller/state_net_trained.pth"))
    parser.add_argument("--legacy-data", type=Path, default=Path("Aebs/data/Downsampled.h5"))
    parser.add_argument("--legacy-policy-state", type=Path, help="Optional full original-SPVC updated AebsEnd2EndNet state_dict")
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--mc-draws", type=int, default=32)
    parser.add_argument("--speed-count", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.spvc_dir is None:
        args.spvc_dir = (Path("results/dino_spvc_minimal_v1") if args.dino_stage == "minimal"
                         else root/"22_R2_original_spvc_scope_compare")
    if args.output_dir is None:
        args.output_dir = (Path("results/dino_spvc_minimal_v1_paired_error") if args.dino_stage == "minimal"
                           else root/"25_legacy_dino_paired_error")
    if args.output_dir.exists() or not 1 <= args.mc_draws <= 128 or min(args.batch_size, args.speed_count) < 2:
        parser.error("new output directory required; draws 1..128, batch/speed-count >=2")
    paths = dict(cache_manifest=args.cache_dir/"cache_manifest.json", features=args.cache_dir/"features.h5",
                 representation=args.representation, dino_controller=args.dino_controller,
                 legacy_controller=args.legacy_controller, legacy_generator=args.legacy_generator,
                 legacy_state_net=args.legacy_state_net, legacy_data=args.legacy_data)
    if args.legacy_policy_state:
        paths["legacy_full_policy"] = args.legacy_policy_state
    if args.dino_stage in ("minimal", "spvc22"):
        paths.update(dino_saved_actor=args.spvc_dir/"latent_ppo_spvc.pt",
                     dino_saved_barrier=args.spvc_dir/"sbc.pt", dino_source_metrics=args.spvc_dir/"metrics.json")
        if args.dino_stage == "minimal":
            paths["dino_final_recheck"] = args.final_recheck
            manifest = args.spvc_dir/"minimal_run_manifest.json"
            if manifest.is_file():
                paths["dino_minimal_manifest"] = manifest
        else:
            paths["dino_scope_comparison"] = args.spvc_dir/"scope_comparison.json"
    else:
        paths["dino_ppo_metrics"] = args.dino_controller.with_name("metrics.json")
    for path in paths.values():
        if not path.is_file():
            parser.error("missing input: " + str(path))
    torch.set_num_threads(1)
    started = time.monotonic()
    hashes = {key: sha256(path) for key, path in paths.items()}
    cache = json.loads(paths["cache_manifest"].read_text())
    rep = torch.load(args.representation, map_location="cpu")
    if (cache.get("schema") != "spvc_expanded_dino_h5_cache_v1" or cache.get("test_encoded") != 0
            or cache["feature_sha256"] != hashes["features"]
            or rep.get("variant") != "R2_group_alignment" or rep.get("dino_frozen") is not True
            or rep["provenance"]["cache_manifest_sha256"] != hashes["cache_manifest"]):
        raise ValueError("R2/cache identities mismatch or test contamination")
    records = cache["records"]
    if any(r["split"] not in ("train", "validation") or r["feature_row"] != i for i, r in enumerate(records)):
        raise ValueError("invalid feature row ordering/split")
    ids = np.asarray([i for i, r in enumerate(records) if r["split"] == args.split], dtype=int)
    if not len(ids):
        raise ValueError("empty selected split")
    rows = [records[i] for i in ids]
    data_root = (args.data_root or Path(cache["data_root"])).resolve()
    with h5py.File(paths["features"], "r") as stream:
        if not stream.attrs["complete"] or stream["features"].shape != (len(records), 384):
            raise ValueError("incomplete/invalid feature cache")
        cached_names = stream["filename"].asstr()[:]
        cached_distances = stream["distance_m"][:]
        if list(cached_names) != [r["filename"] for r in records] or not np.array_equal(
                cached_distances, np.asarray([r["distance_m"] for r in records], dtype=np.float64)):
            raise ValueError("feature cache labels do not match manifest")
        features = np.asarray(stream["features"][ids], dtype=np.float32)
    projection, decoder, q = SafetyProjection(), SafetyDecoder(), PhysicalToLatent()
    for net, key in ((projection,"projection_state_dict"), (decoder,"auxiliary_decoder_state_dict"),
                     (q,"physical_to_latent_state_dict")):
        net.load_state_dict(rep[key], strict=True)
        net.eval()
    scale = float(rep["distance_scale_m"])
    distance = np.asarray([r["distance_m"] for r in rows], dtype=np.float64)
    if scale <= 0 or not np.isfinite(scale) or not np.isfinite(features).all():
        raise ValueError("invalid feature/scale")
    if (not torch.isfinite(rep["feature_mean"]).all() or not torch.isfinite(rep["feature_std"]).all()
            or not (rep["feature_std"] > 0).all()):
        raise ValueError("invalid stored feature normalization")
    with torch.no_grad():
        z_image = projection((torch.from_numpy(features)-rep["feature_mean"])/rep["feature_std"])
        z_q = q(torch.from_numpy((distance[:,None]/scale).astype(np.float32)))
        d_image = decoder(z_image).numpy().flatten()*scale
        d_q = decoder(z_q).numpy().flatten()*scale
    if args.dino_stage in ("minimal", "spvc22"):
        _, saved_actor, _, identities = load_spvc_pair(
            args.representation, args.dino_controller, args.spvc_dir,
            final_recheck=args.final_recheck if args.dino_stage == "minimal" else None)
        dino_net = saved_actor.policy.controller_net.eval()
    else:
        metadata = json.loads(args.dino_controller.with_name("metrics.json").read_text())
        if metadata["representation_sha256"] != hashes["representation"] or metadata["checkpoint_sha256"] != hashes["dino_controller"]:
            raise ValueError("PPO09 identity mismatch")
        dino = PPO.load(args.dino_controller, device="cpu")
        check_ppo(dino, 33)
        dino_net = torch.nn.Sequential(dino.policy.mlp_extractor.policy_net, dino.policy.action_net).eval()
        identities = {"stage": "ppo09", "controller_sha256": hashes["dino_controller"]}
    legacy = PPO.load(args.legacy_controller, device="cpu")
    check_ppo(legacy, 2)
    generator = AebsMLPGenerator(4, 1)
    generator.load_state_dict(torch.load(args.legacy_generator, map_location="cpu"), strict=True)
    old = AebsEnd2EndNet(generator, [1024,256,64,1], legacy.policy.mlp_extractor.policy_net, legacy.policy.action_net)
    old.state_net.load_state_dict(torch.load(args.legacy_state_net, map_location="cpu"), strict=True)
    if args.legacy_policy_state:
        old.load_state_dict(torch.load(args.legacy_policy_state, map_location="cpu"), strict=True)
    old.eval()
    for net in (projection, decoder, q, dino_net, old):
        for parameter in net.parameters():
            parameter.requires_grad_(False)
    with h5py.File(args.legacy_data, "r") as stream:
        legacy_scale = float(np.std(np.asarray(stream["y_train"], dtype=np.float32)))
    if not np.isfinite(legacy_scale) or legacy_scale <= 0:
        raise ValueError("invalid legacy distance scale")
    # Fixed common random numbers across distances/appearances; not best-of-K selection.
    latent_draws = np.random.default_rng(args.seed).uniform(-1,1,(args.mc_draws,4)).astype(np.float32)
    latent_draws = np.concatenate([np.zeros((1,4),dtype=np.float32), latent_draws])
    current = {"dino_image": d_image, "dino_proxy": d_q,
               "legacy_image_01": np.empty(len(rows)), "legacy_image_signed": np.empty(len(rows)),
               "legacy_proxy": np.empty((len(rows),len(latent_draws)))}
    speeds = np.linspace(0, 3, args.speed_count, dtype=np.float32)
    actions = {"dino_image": predict_actions(dino_net,z_image.numpy(),speeds),
               "dino_proxy": predict_actions(dino_net,z_q.numpy(),speeds),
               "legacy_image_01": np.empty((len(rows),len(speeds)),dtype=np.float32),
               "legacy_image_signed": np.empty((len(rows),len(speeds)),dtype=np.float32),
               "legacy_proxy": np.empty((len(rows),len(latent_draws),len(speeds)),dtype=np.float32)}
    args.output_dir.mkdir(parents=True)
    report = dict(status="running", scope="same RGB/state frozen-system development A/B, not a certificate",
                  training=False, test_used=False, split=args.split, hashes=hashes, dino_identity=identities,
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  legacy_scale_m=legacy_scale, dino_scale_m=scale, rgb_hashes_checked=0,
                  legacy_policy_kind="explicit full SPVC state_dict" if args.legacy_policy_state else "shipped base PPO (not final original-SPVC checkpoint)",
                  preprocessing={"primary":"unchanged downSample.py grayscale [0,1], direct state_net",
                                 "sensitivity":"same grayscale mapped to [-1,1]; NOT a model change or selected winner",
                                 "generator":"raw tanh output into state_net, unchanged Combined_network/model.py"},
                  limitations=["existing checkpoints have unequal training data and different controllers",
                               "baseline defaults to shipped base PPO, not a claimed final original-SPVC policy",
                               "validation used for R2 development; not an untouched final test",
                               "DINO distance decoder is auxiliary, not the deployed PPO input",
                               "q and generator consume true distance; proxy distance recovery is not visual estimation",
                               "same-image successor discrepancy is not real CARLA physics model error",
                               "no closed-loop performance or safety claim; repeated speeds/latents/appearances are correlated"])
    write_json(args.output_dir/"metrics.json",report)
    for start in range(0,len(rows),args.batch_size):
        stop = min(start+args.batch_size,len(rows))
        pixels = []
        for row in rows[start:stop]:
            path = (data_root/row["filename"]).resolve()
            if path.parent != data_root:
                raise ValueError("image outside registered root")
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("RGB changed since DINO cache: " + row["filename"])
            with Image.open(io.BytesIO(content)) as image:
                if image.size != (640,640) or image.format != "PNG":
                    raise ValueError("unexpected image format/size")
                pixels.append(legacy_pixels(image).flatten())
        pixels = torch.from_numpy(np.stack(pixels))
        with torch.no_grad():
            for name, x in (("legacy_image_01",pixels),("legacy_image_signed",pixels*2-1)):
                normalized_estimate = old.state_net(x).numpy()
                current[name][start:stop] = normalized_estimate.flatten()*legacy_scale
                actions[name][start:stop] = predict_actions(old.controller_net,normalized_estimate,speeds)
            n, k = stop-start, len(latent_draws)
            y = np.repeat((distance[start:stop,None]/legacy_scale).astype(np.float32),k,axis=0)
            generated = old.gen_net(torch.from_numpy(np.tile(latent_draws,(n,1))),torch.from_numpy(y))
            normalized_estimate = old.state_net(generated.flatten(1)).numpy()
            current["legacy_proxy"][start:stop] = normalized_estimate.reshape(n,k)*legacy_scale
            actions["legacy_proxy"][start:stop] = predict_actions(old.controller_net,normalized_estimate,speeds).reshape(n,k,len(speeds))
            # Check reconstructed generated path against the original composed network at a fixed speed.
            state = torch.from_numpy(np.column_stack([y.flatten(),np.full(n*k,speeds[-1],dtype=np.float32)]))
            direct = old(torch.from_numpy(np.tile(latent_draws,(n,1))),state).numpy().reshape(n,k)
            if not np.allclose(np.clip(direct,-3,3),actions["legacy_proxy"][start:stop,:,-1],rtol=1e-5,atol=1e-5):
                raise ValueError("legacy composed-path equivalence failed")
        print("paired RGB %d/%d" % (stop,len(rows)),flush=True)
    if any(not np.isfinite(x).all() for x in current.values()):
        raise ValueError("nonfinite current-distance prediction")
    valid = np.asarray([[classify_terminal_outcome(d,float(v)) is None for v in speeds] for d in distance])
    groups = np.asarray([r["distance_bin_025m"] for r in rows])
    dino_delta = next_speed(speeds,actions["dino_proxy"])-next_speed(speeds,actions["dino_image"])
    by_scope = {"all_selected": np.ones(len(rows),dtype=bool)}
    for key in sorted({r["weather"]+"|"+r["color"] for r in rows}):
        by_scope[key] = np.asarray([r["weather"]+"|"+r["color"] == key for r in rows])
    comparison = {}
    for label, mask in by_scope.items():
        c = {"images": int(mask.sum()), "current_distance": {}, "next_speed": {}}
        for mode in ("01","signed"):
            c["current_distance"]["image_"+mode] = paired_summary(
                d_image[mask]-distance[mask],current["legacy_image_"+mode][mask]-distance[mask],groups[mask])
            old_delta = next_speed(speeds,actions["legacy_proxy"][mask])-next_speed(speeds,actions["legacy_image_"+mode][mask,None,:])
            for kind, sl in (("z_zero",slice(0,1)),("uniform_mc",slice(1,None))):
                c["next_speed"][mode+"_"+kind] = transition_summary(dino_delta[mask],old_delta[:,sl,:],valid[mask],groups[mask])
        for kind, sl in (("z_zero",slice(0,1)),("uniform_mc",slice(1,None))):
            old_error = current["legacy_proxy"][mask,sl]-distance[mask,None]
            new_error = np.broadcast_to((d_q[mask]-distance[mask])[:,None],old_error.shape)
            c["current_distance"]["proxy_"+kind] = paired_summary(new_error,old_error,groups[mask])
        comparison[label] = c
    if {key: sha256(path) for key, path in paths.items()} != hashes:
        raise ValueError("model/cache/evidence input changed during evaluation; fixed-pair result invalid")
    report.update(status="development_comparison_complete",input_files_unchanged=True,
                  rgb_hashes_checked=len(rows),images=len(rows),
                  nonterminal_state_pairs=int(valid.sum()),excluded_terminal_state_pairs=int(valid.size-valid.sum()),
                  comparison=comparison,
                  dino_image_proxy_distance_gap_m=errors(d_q-d_image),
                  legacy_image_proxy_distance_gap_m={mode:errors(current["legacy_proxy"][:,1:]-current["legacy_image_"+mode][:,None]) for mode in ("01","signed")},
                  runtime_seconds=time.monotonic()-started)
    np.savez_compressed(args.output_dir/"samples.npz",distance_m=distance,speeds_mps=speeds,nonterminal=valid,
                        feature_row=ids,filename=np.asarray([r["filename"] for r in rows]),
                        distance_block=groups,latent_draws=latent_draws,
                        **{"distance_"+k:v for k,v in current.items()},**{"action_"+k:v for k,v in actions.items()})
    write_json(args.output_dir/"metrics.json",report)
    print("\n[Legacy cGAN vs DINO: same RGB/current state — development only]")
    print("images=%d nonterminal_pairs=%d DINO=%s; no training/test/rollout" % (len(rows),valid.sum(),args.dino_stage))
    print("legacy:",report["legacy_policy_kind"])
    if args.dino_stage == "minimal":
        print("final saved actor SHA256:", identities["new_controller_sha256"])
        print("checkpoint matches final recheck:", identities["checkpoint_matches_final_recheck"])
    for family in ("current_distance","next_speed"):
        for name,m in comparison["all_selected"][family].items():
            print("%s %s: legacy MAE=%.9f RMSE=%.9f | DINO MAE=%.9f RMSE=%.9f | lower=%s" % (
                family,name,m["legacy"]["mae"],m["legacy"]["rmse"],m["dino"]["mae"],m["dino"]["rmse"],m["sampled_mae_and_rmse_lower"]))
    print("Distance errors: m; successor-speed errors: m/s. Lower is sampled only, not a paper/safety claim.")
    print("metrics:",args.output_dir/"metrics.json")


if __name__ == "__main__":
    main()
