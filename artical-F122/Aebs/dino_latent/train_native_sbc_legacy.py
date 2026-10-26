"""Barrier-only legacy-SPVC baseline for the frozen NEW native R2 PPO.

No CP residual box is included and no controller optimization is called.
Original VT losses/noise/grid/filter/interval implementation are preserved.
The resulting readout is explicitly a legacy diagnostic, not image safety.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from Aebs.system.env import Aebs
from Aebs.VT.train import VTLearner
from Aebs.VT.verify import VTVerifier
from Aebs.dino_latent.native_sbc_stage import load_frozen
from Aebs.dino_latent.prepare_expanded_data import sha256


class NativeVTPolicy(nn.Module):
    def __init__(self, physical_policy, state_scale):
        super().__init__()
        self.q_model = physical_policy.q
        self.controller_net = physical_policy.actor
        self.register_buffer("distance_mean", physical_policy.distance_mean.detach().clone())
        self.register_buffer("distance_std", physical_policy.distance_std.detach().clone())
        self.state_scale = float(state_scale)
        self.requires_grad_(False); self.eval()

    def controller_input(self, unused, normalized_state):
        x = (normalized_state[:, :1] * self.state_scale - self.distance_mean) / self.distance_std
        return torch.cat((self.q_model(x), normalized_state[:, 1:2]), 1)

    def forward(self, unused, normalized_state):
        return self.controller_net(self.controller_input(unused, normalized_state)).clamp(-3, 3)


def check(verifier, grid):
    # Evaluation snapshot avoids freezing/training a stale or different model.
    verifier.learner.l_model.eval()
    verifier.l_ibp = verifier.learner.create_bounded_module(copy.deepcopy(verifier.learner.l_model))
    sat, hard, info, _ = verifier.check_dec_cond(1.2)
    _, upper = verifier.compute_bound_init(grid)
    lower, _ = verifier.compute_bound_unsafe(grid)
    minimum, _ = verifier.compute_bound_domain(grid)
    states, _, _ = verifier.get_unfiltered_grid(grid)
    with torch.no_grad():
        values = verifier.learner.l_model(torch.as_tensor(states, dtype=torch.float32, device=verifier.device)).flatten()
    normalized = (values - minimum) / max(upper - minimum, 1e-6)
    normalized_unsafe = (lower - minimum) / max(upper - minimum, 1e-6)
    checked = int(((normalized > .95 * normalized_unsafe) & (normalized < normalized_unsafe)).sum())
    separated = bool(np.isfinite([upper, lower, minimum]).all() and lower > upper > minimum)
    readout = float(1 - (upper - minimum) / (lower - minimum)) if separated else None
    return dict(original_threshold_pass=bool(sat), hard_violations=int(hard), grid_states=len(states),
                actual_checked_band_states=checked, init_upper=upper, unsafe_lower=lower, domain_lower=minimum,
                separated=separated, legacy_probability_readout=readout, zero_hard_violations=int(hard) == 0,
                formal_certificate=False, image_proxy_error_included=False, info=info)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-dir", type=Path, default=Path("results/native_representation_dimensions_20261019_v1"))
    p.add_argument("--controller-dir", type=Path, default=Path("results/native_controllers_speed_residual_20261019_v1/historical_R2"))
    p.add_argument("--rounds", type=int, default=80)
    p.add_argument("--epochs-per-round", type=int, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() or min(args.rounds, args.epochs_per_round) < 1:
        p.error("New output and positive budgets required")
    torch.set_num_threads(1); torch.manual_seed(7); np.random.seed(7)
    physical_policy, _, metadata = load_frozen(args.native_dir, args.controller_dir)
    backend_paths = [Path("Aebs/VT/train.py"), Path("Aebs/VT/verify.py"), Path("Aebs/VT/utils.py"), Path("Aebs/system/env.py")]
    metadata["backend_sha256"] = {str(path): sha256(path) for path in backend_paths}
    env = Aebs(.05)
    policy = NativeVTPolicy(physical_policy, env.std1)
    sample = torch.tensor([[15., 2.5], [7., 1.], [6.1, .3]])
    with torch.no_grad():
        normalized = sample.clone(); normalized[:, 0] /= env.std1
        equivalence = float((policy(None, normalized) - physical_policy(sample)).abs().max())
    if equivalence > 1e-5: raise ValueError("VT adapter coordinate mismatch")
    learner = VTLearner([2, 16, 8, 1], env, 2., 4., .1, 1., .95,
                        square_l_output=True, p_net=policy)
    before = {k: v.detach().cpu().clone() for k, v in learner.p_net.state_dict().items()}
    initial_barrier = {k: v.detach().cpu().clone() for k, v in learner.l_model.state_dict().items()}
    verifier = VTVerifier(learner, env, None, 2048, .9, False)
    verifier.prefill_train_buffer()
    train_states, _, _ = verifier.get_unfiltered_grid(env.train_space_split)
    delta = (env.observation_space.high - env.observation_space.low) / env.train_space_split
    args.output_dir.mkdir(parents=True)
    config = dict(rounds=args.rounds, epochs_per_round=args.epochs_per_round, seed=7,
                  policy_frozen=True, CP_error_included=False,
                  no_actor_update=True, grid=[100, 100], original_noise_factor=.05,
                  legacy_threshold=.001, original_VT_backend_unchanged=True,
                  evaluation_wrapper="fresh frozen deep-copy of current barrier each check",
                  scope="legacy assumed-noise proxy SBC baseline only, NOT calibrated image-path guarantee",
                  adapter_action_max_abs=metadata["adapter_action_max_abs"], VT_adapter_max_abs=equivalence,
                  provenance=metadata, source_sha256=sha256(Path(__file__)))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    history = []; best_key = None; best_state = None; started = time.monotonic()
    for iteration in range(1, args.rounds + 1):
        learner.l_model.train()
        metrics = learner.train_epoch(train_states, delta, lip=.001, batch_size=2048,
                                      shuffle=True, num_epochs=args.epochs_per_round, train_fn="l")
        verify = check(verifier, env.space_split)
        record = dict(round=iteration, final_training=metrics[-1], verification=verify)
        history.append(record)
        key = (verify["separated"] and verify["actual_checked_band_states"] > 0,
               -verify["hard_violations"], verify["legacy_probability_readout"] or -1e10)
        if best_key is None or key > best_key:
            best_key = key; best_state = copy.deepcopy(learner.l_model.state_dict()); best_round = iteration
            torch.save(best_state, args.output_dir / "barrier_best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, default=float))
        print("[Frozen-policy SBC] round=%d band=%d hard=%d separation=%s legacy_readout=%s" % (
            iteration, verify["actual_checked_band_states"], verify["hard_violations"],
            verify["separated"], verify["legacy_probability_readout"]), flush=True)
        if verify["zero_hard_violations"] and verify["separated"] and verify["actual_checked_band_states"] > 0 and verify["legacy_probability_readout"] >= .9:
            break
    torch.save(learner.l_model.state_dict(), args.output_dir / "barrier_last.pt")
    learner.l_model.load_state_dict(best_state)
    torch.save(learner.l_model.state_dict(), args.output_dir / "sbc.pt")
    final = check(verifier, env.space_split)
    if any(not torch.equal(v.detach().cpu(), before[k]) for k, v in learner.p_net.state_dict().items()):
        raise ValueError("Frozen actor/q changed during SBC-only training")
    changed = sum(not torch.equal(v.detach().cpu(), initial_barrier[k]) for k, v in learner.l_model.state_dict().items())
    if not changed: raise ValueError("Barrier did not train")
    for path, digest in metadata["source_sha256"].items():
        if sha256(Path(path)) != digest: raise ValueError("Frozen source changed")
    for path, digest in metadata["backend_sha256"].items():
        if sha256(Path(path)) != digest: raise ValueError("VT backend changed")
    result = dict(config=config, status="legacy_proxy_checks_pass" if final["original_threshold_pass"] and final["separated"] and final["actual_checked_band_states"] > 0 else "legacy_proxy_checks_not_passed",
                  best_round=best_round, rounds_completed=len(history), verification=final,
                  checkpoint_matches_verification=True, controller_q_tensorwise_unchanged=True,
                  barrier_changed_tensors=changed, runtime_seconds=time.monotonic() - started,
                  saved_barrier_sha256=sha256(args.output_dir / "sbc.pt"), formal_certificate=False,
                  image_path_guarantee=False, original_backend_unchanged=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print("[Frozen new PPO legacy SBC baseline]", result["status"], final, flush=True)
    print("metrics:", args.output_dir / "metrics.json", flush=True)


if __name__ == "__main__": main()
