"""Read-only final-checkpoint check with the existing, unmodified VTVerifier.

No optimizer, training, new scope, corrected interval implementation or CP/QP.
Writes a separate report, never overwrites the training run or its checkpoints.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def summarize(hard, denominator, selected, compensated, init, unsafe, minimum, returned_pass):
    if denominator <= 0 or not 0 <= hard <= selected <= denominator or not hard <= compensated <= selected:
        raise ValueError("invalid grid/violation counts")
    fraction = hard / denominator
    if bool(returned_pass) != (fraction <= .001):
        raise ValueError("original verifier return does not match its recorded threshold")
    finite = all(math.isfinite(x) for x in (init,unsafe,minimum))
    valid_regions = finite and init > minimum and unsafe >= init
    probability = 1-(init-minimum)/(unsafe-minimum) if valid_regions else None
    if not finite:
        status = "invalid_nonfinite_region_bounds"
    elif selected == 0:
        status = "empty_legacy_band_not_evidence"
    elif not returned_pass:
        status = "saved_pair_fails_legacy_decrease_threshold"
    elif probability is None or probability < .9:
        status = "saved_pair_does_not_reach_legacy_probability_target"
    else:
        status = "saved_pair_passes_legacy_checks"
    return dict(status=status, hard_violations=hard, compensated_violations=compensated,
                grid_denominator=denominator, actually_checked_band_states=selected,
                hard_fraction_total_grid=fraction,
                hard_fraction_checked_band=hard/selected if selected else None,
                legacy_threshold_pass=bool(returned_pass), zero_hard_violations=hard == 0,
                init_upper=init if finite else None, unsafe_lower=unsafe if finite else None,
                domain_lower=minimum if finite else None, legacy_probability_readout=probability,
                probability_target=.9, end_to_end_safety_certificate=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir",type=Path,default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--output-dir",type=Path,default=Path("results/dino_spvc_minimal_v1_final_recheck"))
    parser.add_argument("--representation-checkpoint",type=Path,default=None,help="Override a moved source path; hash must match")
    parser.add_argument("--controller",type=Path,default=None,help="Override a moved base PPO path; hash must match")
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve() == args.checkpoint_dir.resolve():
        parser.error("use a new output directory, separate from the saved run")
    paths = {"metrics":args.checkpoint_dir/"metrics.json", "barrier":args.checkpoint_dir/"sbc.pt",
             "policy":args.checkpoint_dir/"latent_ppo_spvc.pt"}
    for path in paths.values():
        if not path.is_file():
            parser.error("missing input: " + str(path))
    source = json.loads(paths["metrics"].read_text())
    if source.get("experiment") != "R2_DINO_original_SPVC_front_end_only_v1" or source.get("test_used") is not False:
        parser.error("not a supported original-method R2 checkpoint pair")
    paths["representation"] = args.representation_checkpoint or Path(source["representation_checkpoint"])
    paths["base_controller"] = args.controller or Path(source["controller_checkpoint"])
    if any(not p.is_file() for p in paths.values()):
        parser.error("missing source representation or base PPO; supply relocated paths explicitly")
    hashes = {key:digest(path) for key,path in paths.items()}
    for key,field in (("barrier","saved_barrier_sha256"),("policy","saved_policy_sha256"),
                      ("representation","representation_sha256"),("base_controller","controller_sha256")):
        if hashes[key] != source.get(field):
            parser.error("checkpoint identity mismatch: " + key)
    expected = {"environment_factor":.05,"l_model_config":[2,16,8,1],"p_lip":2.,"l_lip":4.,
                "epsilon":.1,"gamma_decrease":1.,"learner_reach_prob":.95,"verifier_reach_prob":.9,
                "grid":[100,100],"hard_violation_fraction_threshold":.001,
                "interval_implementation":"legacy VTVerifier unchanged"}
    if any(source.get("original_vt_settings",{}).get(k) != v for k,v in expected.items()):
        parser.error("saved settings differ from the supported minimal original-SPVC run")
    aebs_root = Path(__file__).resolve().parents[1]
    backend = {name:digest(aebs_root/name) for name in ("VT/train.py","VT/loop.py","VT/verify.py","VT/utils.py","system/env.py")}
    manifest_path = args.checkpoint_dir/"minimal_run_manifest.json"
    manifest_checked = manifest_path.is_file()
    if manifest_checked:
        manifest = json.loads(manifest_path.read_text())
        paths["minimal_manifest"] = manifest_path
        hashes["minimal_manifest"] = digest(manifest_path)
        if manifest.get("backend_hashes_after") != backend or manifest.get("backend_files_unchanged_during_run") is not True:
            parser.error("backend source differs from training run; do not silently change the verification method")

    import numpy as np
    import torch
    from auto_LiRPA import BoundedModule
    from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
    from Aebs.system.env import Aebs
    from Aebs.VT.utils import MLP
    from Aebs.VT.verify import VTVerifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DinoLatentSPVCPolicy.from_checkpoints(paths["representation"],paths["base_controller"],device)
    frozen_q = {k:v.detach().clone() for k,v in policy.q_model.state_dict().items()}
    policy.load_state_dict(torch.load(paths["policy"],map_location=device),strict=True)
    if any(not torch.equal(v,frozen_q[k]) for k,v in policy.q_model.state_dict().items()):
        raise ValueError("saved q differs from the frozen representation")
    saved_scale = float(source["policy"]["state_distance_scale_m"])
    policy.set_state_distance_scale(saved_scale)
    env = Aebs(.05)
    if not np.isclose(env.std1,saved_scale,rtol=1e-6,atol=1e-7) or not np.array_equal(env.space_split,[100,100]):
        raise ValueError("current VT coordinates/grid differ from the saved run")
    barrier = MLP([2,16,8,1],activation="tanh",square_output=True).to(device)
    barrier.load_state_dict(torch.load(paths["barrier"],map_location=device),strict=True)
    for model in (policy,barrier):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    learner = SimpleNamespace(p_net=policy,l_model=barrier)
    bounded = BoundedModule(barrier,torch.zeros((1,2),device=device),device=device)
    verifier = VTVerifier(learner,env,bounded,batch_size=2048,reach_prob=.9,fail_check_fast=True)
    verifier.prefill_train_buffer()  # Preserve original threshold denominator (10000).
    observed = {"selected":0,"compensated":0,"hard":0,"states":[]}
    original_check_batch = verifier._check_dec_batch
    original_expectation = verifier.compute_expected_l

    def finite_expectation(*values):
        result = original_expectation(*values)
        if not torch.isfinite(result).all():
            raise ValueError("nonfinite legacy expectation; cannot report a passing result")
        return result

    verifier.compute_expected_l = finite_expectation

    def record_batch(s,a,l,K):
        # Logging only: the original method receives the exact inputs and its
        # exact return value is passed back to check_dec_cond without changes.
        result = original_check_batch(s,a,l,K)
        if not torch.isfinite(l).all() or not torch.isfinite(a).all():
            raise ValueError("nonfinite saved-policy/barrier output")
        observed["selected"] += len(s)
        observed["compensated"] += int(result[0])
        observed["hard"] += int(result[2])
        observed["states"].append(s.detach().cpu().numpy())
        return result

    verifier._check_dec_batch = record_batch
    passed, hard, check_info, _ = verifier.check_dec_cond(1.2)
    if int(hard) != observed["hard"]:
        raise ValueError("logged hard count differs from original verifier")
    _, init = verifier.compute_bound_init(env.space_split)
    unsafe, _ = verifier.compute_bound_unsafe(env.space_split)
    minimum, _ = verifier.compute_bound_domain(env.space_split)
    outcome = summarize(int(hard),len(verifier.train_buffer),observed["selected"],observed["compensated"],
                        float(init),float(unsafe),float(minimum),bool(passed))
    after = {key:digest(path) for key,path in paths.items()}
    backend_after = {name:digest(aebs_root/name) for name in backend}
    if hashes != after or backend != backend_after:
        raise ValueError("input files changed during verification; result not attributable to a fixed pair")
    result = dict(experiment="minimal_R2_saved_pair_original_SPVC_recheck",training_run=False,test_used=False,
                  checkpoint_matches_verification=True,source_dir=str(args.checkpoint_dir),source_hashes=hashes,
                  backend_hashes=backend,training_backend_manifest_checked=manifest_checked,
                  checkpoint_files_unchanged=True,settings=expected,K_except_barrier=1.2,
                  noise_cells=int(np.prod(verifier.pmass_n)),original_fail_check_fast=True,
                  original_check_info=check_info,source_last_loop_info=source.get("loop_info"),
                  vt_data_sha256=digest(Path("Aebs/data/Downsampled.h5")),
                  limitations=["same legacy narrow-band and interval implementation, not corrected or full-domain",
                               "probability readout uses the original region ratio, not a new proof",
                               "DINO image/q discrepancy is not formally covered",
                               "checkpoint match denotes identity, not verification success",
                               "training did not record a VT-data hash; distance scale was checked"],**outcome)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"metrics.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    states = np.concatenate(observed["states"]) if observed["states"] else np.empty((0,2),dtype=np.float32)
    np.savez_compressed(args.output_dir/"checked_band_states.npz",states=states)
    print("\n[Minimal DINO-SPVC final saved-pair recheck — original verifier, no training]")
    print("status:",outcome["status"])
    print("checkpoint matches verification: True")
    print("actual checked band states: %d/%d" % (observed["selected"],len(verifier.train_buffer)))
    print("hard violations: %d/%d; compensated=%d" % (int(hard),len(verifier.train_buffer),observed["compensated"]))
    print("original <=0.1%% threshold: %s; zero hard violations: %s" % (bool(passed),int(hard)==0))
    print("init upper=%.9f unsafe lower=%.9f domain lower=%.9f" % (init,unsafe,minimum))
    print("legacy probability readout:",outcome["legacy_probability_readout"])
    print("Not an end-to-end image-path certificate. Checkpoint files unchanged.")
    print("metrics:",args.output_dir/"metrics.json")


if __name__ == "__main__":
    main()
