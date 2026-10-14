"""Inference-only adapter for the actor saved by run_spvc (not a PPO resume file)."""

import json

import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy
from Aebs.dino_latent.saved_pair_provenance import check_final_recheck


class SavedSpvcActor:
    def __init__(self, policy, low, high):
        self.policy = policy.eval()
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        if not deterministic:
            raise ValueError("SPVC saves an actor mean, not a retrained PPO distribution")
        obs = np.asarray(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        if obs.ndim != 2 or obs.shape[1] != self.policy.latent_dimension + 1 or not np.isfinite(obs).all():
            raise ValueError("expected finite latent+speed observations")
        with torch.no_grad():
            action = self.policy.controller_net(torch.from_numpy(obs)).cpu().numpy()
        if not np.isfinite(action).all():
            raise ValueError("nonfinite actor output")
        action = np.clip(action, self.low, self.high)
        return (action[0] if single else action), None


def load_spvc_pair(representation, base_controller, spvc_dir, final_recheck=None):
    metadata = json.loads((spvc_dir / "metrics.json").read_text(encoding="utf-8"))
    saved_path = spvc_dir / "latent_ppo_spvc.pt"
    barrier_path = spvc_dir / "sbc.pt"
    if (metadata.get("experiment") != "R2_DINO_original_SPVC_front_end_only_v1"
            or metadata.get("test_used") is not False
            or metadata.get("representation_sha256") != sha256(representation)
            or metadata.get("controller_sha256") != sha256(base_controller)
            or metadata.get("saved_policy_sha256") != sha256(saved_path)
            or metadata.get("saved_barrier_sha256") != sha256(barrier_path)):
        raise ValueError("SPVC model hashes/provenance do not match; do not evaluate another PPO")
    if final_recheck is not None:
        evidence = check_final_recheck(representation, base_controller, spvc_dir, final_recheck)
    else:
        scope_path = spvc_dir / "scope_comparison.json"
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        if any(scope.get(key) != metadata[key] for key in (
            "saved_policy_sha256", "saved_barrier_sha256"
        )):
            raise ValueError("scope comparison and saved checkpoint hashes disagree")
        evidence = {"stage": "spvc22", "scope_comparison_sha256": sha256(scope_path)}
    old = PPO.load(base_controller, device="cpu")
    old.policy.set_training_mode(False)
    if (old.observation_space.shape != (33,) or old.action_space.shape != (1,)
            or not np.array_equal(old.action_space.low, [-3.0])
            or not np.array_equal(old.action_space.high, [3.0])
            or getattr(old.policy, "squash_output", False)):
        raise ValueError("unsupported PPO input/action interface")
    adapted = DinoLatentSPVCPolicy.from_checkpoints(representation, base_controller, torch.device("cpu"))
    reference = {key: value.detach().clone() for key, value in adapted.state_dict().items()}
    # strict loading is intentional: no silent fallback to the base actor.
    adapted.load_state_dict(torch.load(saved_path, map_location="cpu"), strict=True)
    if any(not torch.equal(value, reference[key]) for key, value in adapted.state_dict().items()
           if key.startswith("q_model.")):
        raise ValueError("SPVC changed the supposedly frozen q representation")
    info = metadata["policy"]
    if float(info["distance_scale_m"]) != adapted.distance_scale_m:
        raise ValueError("representation distance scale mismatch")
    adapted.set_state_distance_scale(info["state_distance_scale_m"])
    adapted.eval()
    old_adapter = DinoLatentSPVCPolicy.from_checkpoints(representation, base_controller, torch.device("cpu"))
    changed = sum(not torch.equal(value, reference[key]) for key, value in adapted.state_dict().items()
                  if key.startswith("controller_net."))
    return old, SavedSpvcActor(adapted, old.action_space.low, old.action_space.high), \
        SavedSpvcActor(old_adapter, old.action_space.low, old.action_space.high), {
            "old_controller_sha256": sha256(base_controller),
            "new_controller_sha256": sha256(saved_path),
            "barrier_sha256": sha256(barrier_path),
            "source_metrics_sha256": sha256(spvc_dir / "metrics.json"),
            **evidence,
            "changed_actor_tensors": changed,
            "q_parameters_unchanged": True,
            "state_distance_scale_m": adapted.state_distance_scale_m,
            "representation_distance_scale_m": adapted.distance_scale_m,
            "inference": "deterministic actor mean clipped to [-3,3]; no QP; no SBC action filtering",
        }


def check_adapter_equivalence(old, new, old_adapter, image_obs, q_obs, distance, speed):
    """Separate same-input loading checks from float32 coordinate roundoff.

    d / R2_scale and (d / VT_scale) * (VT_scale / R2_scale) need not be
    bit-identical. Check their INPUT discrepancy against a small floating-point
    budget; do not interpret the network-amplified output difference as a load
    failure. Record that difference, without changing inference or model weights.
    """
    errors = {}
    for name, obs in (("image", image_obs), ("q", q_obs)):
        baseline = old.predict(obs, deterministic=True)[0]
        reconstructed = old_adapter.predict(obs)[0]
        errors["old_adapter_%s_max_abs" % name] = float(np.max(np.abs(baseline-reconstructed)))
        with torch.no_grad():
            direct = new.policy.forward_latent(
                torch.from_numpy(obs[:, :-1]), torch.from_numpy(obs[:, -1:])
            ).numpy()
        errors["new_adapter_%s_max_abs" % name] = float(np.max(np.abs(
            np.clip(direct, -3, 3)-new.predict(obs)[0]
        )))
    states = np.column_stack([distance / new.policy.state_distance_scale_m, speed]).astype(np.float32)
    state_tensor = torch.from_numpy(states)
    direct_q_distance = torch.from_numpy(
        np.asarray(distance[:, None] / new.policy.distance_scale_m, dtype=np.float32)
    )
    vt_q_distance = state_tensor[:, :1] * (
        new.policy.state_distance_scale_m / new.policy.distance_scale_m
    )
    with torch.no_grad():
        z = torch.zeros((len(states), 4))
        vt = new.policy(z, state_tensor).numpy()
        actual_vt_obs = new.policy.controller_input(z, state_tensor).numpy()
        reconstructed_vt_obs = torch.cat([
            new.policy.q_model(vt_q_distance), state_tensor[:, 1:2]
        ], dim=1).numpy()
        reconstructed_physical_obs = torch.cat([
            new.policy.q_model(direct_q_distance), torch.from_numpy(speed[:, None])
        ], dim=1).numpy()
    errors["new_vt_same_input_max_abs"] = float(np.max(np.abs(
        np.clip(vt, -3, 3)-new.predict(actual_vt_obs)[0]
    )))
    errors["vt_observation_reconstruction_max_abs"] = float(np.max(np.abs(
        actual_vt_obs-reconstructed_vt_obs
    )))
    errors["physical_q_observation_reconstruction_max_abs"] = float(np.max(np.abs(
        q_obs-reconstructed_physical_obs
    )))
    # The same-input tolerance is unchanged. Wrong actor, q input or scaling
    # implementation still fails; only the cross-coordinate test is separated.
    if not all(np.isfinite(value) and value <= 1e-5 for value in errors.values()):
        raise ValueError("actor loading/action equivalence check failed: %s" % errors)
    difference = torch.abs(vt_q_distance-direct_q_distance)
    # Operational roundoff check, not a formal floating-point safety bound.
    budget = 8 * torch.finfo(torch.float32).eps * torch.maximum(
        torch.ones_like(direct_q_distance), torch.abs(direct_q_distance)
    )
    if (not torch.isfinite(difference).all().item()
            or not (difference <= budget).all().item()):
        raise ValueError("VT/physical q input difference exceeds float32 roundoff budget")
    cross_action_error = float(np.max(np.abs(np.clip(vt, -3, 3)-new.predict(q_obs)[0])))
    if not np.isfinite(cross_action_error):
        raise ValueError("nonfinite cross-coordinate action difference")
    errors.update({
        "new_vt_q_vs_physical_q_max_abs": cross_action_error,
        "q_normalized_input_roundoff_max_abs": float(difference.max()),
        "q_normalized_input_roundoff_max_budget_ratio": float((difference/budget).max()),
        "cross_coordinate_action_gap_exceeds_1e_minus5": cross_action_error > 1e-5,
        "check_version": "v2_same_input_strict_coordinate_roundoff_separate",
    })
    if cross_action_error > 1e-5:
        print("[Step23 coordinate roundoff] same-input checks passed; "
              "q-input max difference=%.9g; cross-coordinate action max difference=%.9g. "
              "Recorded as numerical sensitivity, NOT a safety tolerance." % (
                  float(difference.max()), cross_action_error
              ), flush=True)
    return errors
