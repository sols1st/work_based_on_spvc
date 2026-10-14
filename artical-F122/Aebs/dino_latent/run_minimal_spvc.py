"""Minimal frozen semantic/latent front-end substitution into existing SPVC.

This launcher does not implement a new learner, loss, verifier or safety rule.
Only existing R2 and PPO checkpoints are used; no SBC warm start or new scopes.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("results/dino_expanded_v1")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-checkpoint", type=Path,
                        default=ROOT/"08_R2_group_alignment/dino_safety_latent.pt")
    parser.add_argument("--controller", type=Path, default=ROOT/"09_R2_ppo/latent_ppo.zip")
    parser.add_argument("--controller-metrics", type=Path, default=ROOT/"09_R2_ppo/metrics.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dino_spvc_minimal_v1"))
    parser.add_argument("--timeout-seconds", type=float, default=3600.)
    parser.add_argument("--max-iteration-index", type=int, default=100)
    parser.add_argument("--check-only", action="store_true", help="Interface/hash/gradient checks only; no training or output directory")
    return parser


def training_arguments(args):
    # Whitelist only model locations and compute budget. No extra experiment flags.
    names = ("representation_checkpoint", "controller", "controller_metrics", "output_dir",
             "timeout_seconds", "max_iteration_index")
    return [item for name in names for item in ("--"+name.replace("_", "-"), str(getattr(args,name)))]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_interfaces(args):
    import numpy as np
    import torch
    from Aebs.dino_latent.models import SafetyProjection
    from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
    from Aebs.system.env import Aebs

    paths = [args.representation_checkpoint,args.controller,args.controller_metrics,Path("Aebs/data/Downsampled.h5")]
    for path in paths:
        if not path.is_file():
            raise ValueError("missing input: " + str(path))
    rep = torch.load(args.representation_checkpoint,map_location="cpu")
    metrics = json.loads(args.controller_metrics.read_text())
    if (rep.get("variant") != "R2_group_alignment" or rep.get("dino_frozen") is not True
            or rep.get("latent_dimension") != 32 or metrics.get("test_used") is not False
            or metrics.get("representation_sha256") != file_hash(args.representation_checkpoint)
            or metrics.get("checkpoint_sha256") != file_hash(args.controller)):
        raise ValueError("R2/PPO identities or latent dimension mismatch")
    policy = DinoLatentSPVCPolicy.from_checkpoints(args.representation_checkpoint,args.controller,torch.device("cpu"))
    policy.eval()
    env = Aebs(.05)
    policy.set_state_distance_scale(env.std1)
    state = torch.tensor([[15./env.std1,2.5],[16./env.std1,3.],[10./env.std1,1.5]],dtype=torch.float32)
    unused_z = torch.zeros(3,4)
    obs = policy.controller_input(unused_z,state)
    action = policy(unused_z,state)
    if obs.shape != (3,33) or action.shape != (3,1) or not torch.isfinite(action).all():
        raise ValueError("bad SPVC policy interface")
    with torch.no_grad():
        expected_q = policy.q_model(state[:,:1]*(env.std1/policy.distance_scale_m))
        if not torch.allclose(obs[:,:32],expected_q,atol=1e-6,rtol=1e-6):
            raise ValueError("VT distance conversion mismatch")
        if not torch.equal(action,policy.forward_latent(obs[:,:32],state[:,1:2])):
            raise ValueError("physical/latent actor interface mismatch")
        projection = SafetyProjection()
        projection.load_state_dict(rep["projection_state_dict"],strict=True)
        projection.eval()
        # Synthetic normalized DINO features, NOT a real image accuracy test.
        image_latent = projection(torch.zeros(3,384))
        image_action = policy.forward_latent(image_latent,state[:,1:2])
        if image_latent.shape != (3,32) or not torch.isfinite(image_action).all():
            raise ValueError("bad image-projection/actor interface")
    action.sum().backward()
    actor_grad = sum(float(p.grad.abs().sum()) for p in policy.controller_net.parameters() if p.grad is not None)
    q_frozen = all(not p.requires_grad and p.grad is None for p in policy.q_model.parameters())
    if not np.isfinite(actor_grad) or actor_grad <= 0 or not q_frozen:
        raise ValueError("PPO must receive gradients and q must stay frozen")
    policy.zero_grad(set_to_none=True)
    return {"interface_check": "passed", "training_run": False, "real_image_test": False,
            "controller_input_dimension": 33, "SBC_state_dimension": 2,
            "q_frozen": q_frozen, "actor_gradient_nonzero": True,
            "representation_sha256": file_hash(args.representation_checkpoint),
            "controller_sha256": file_hash(args.controller), "policy": policy.metadata()}


def main(argv=None):
    import math
    parser = make_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.max_iteration_index < 0:
        parser.error("positive finite timeout and nonnegative iteration index required")
    if not args.check_only and args.output_dir.exists():
        parser.error("choose a new output directory; existing runs are never overwritten")
    try:
        checks = check_interfaces(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print("[Minimal latent-SPVC interface check]")
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    if args.check_only:
        return
    from Aebs.dino_latent.run_spvc import main as run_original_adapter
    code_root = Path(__file__).resolve().parents[2]
    backend_files = ["VT/train.py","VT/loop.py","VT/verify.py","VT/utils.py","system/env.py"]
    # Hash current repository legacy modules; this is not an upstream-originality claim.
    before = {name:file_hash(code_root/"Aebs"/name) for name in backend_files}
    run_original_adapter(training_arguments(args))
    after = {name:file_hash(code_root/"Aebs"/name) for name in backend_files}
    record = {"mode":"minimal_frontend_substitution", "interface_checks":checks,
              "backend_hashes_before":before,"backend_hashes_after":after,
              "backend_files_unchanged_during_run":before == after,
              "learner_verifier":"existing repository VTLearner / VTVerifier / Loop",
              "extra_scope_checks":False,"CP":False,"QP":False,"residual_contract":False,
              "SBC_warm_start":False,"zero_violation_early_stop":False,
              "claim":"legacy proxy experiment; image/q mismatch is not certified",
              "warning":"original loop info can precede the last actor update; do not label the final weights certified"}
    (args.output_dir/"minimal_run_manifest.json").write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding="utf-8")
    print("[Minimal latent-SPVC finished]")
    print("Reused original repository learner/losses/verifier; no extra experiment enabled.")
    print("Backend files unchanged:",before == after)
    print("Completion is not a safety certificate; read metrics.json checkpoint_matches_verification.")
    print("results:",args.output_dir)


if __name__ == "__main__":
    main()
