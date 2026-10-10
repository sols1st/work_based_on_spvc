"""Inference-only adapter for the actor saved by run_spvc (not a PPO resume file)."""

import json

import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.prepare_expanded_data import sha256
from Aebs.dino_latent.spvc_policy import DinoLatentSPVCPolicy


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


def load_spvc_pair(representation, base_controller, spvc_dir):
    metadata = json.loads((spvc_dir / "metrics.json").read_text(encoding="utf-8"))
    saved_path = spvc_dir / "latent_ppo_spvc.pt"
    barrier_path = spvc_dir / "sbc.pt"
    if (metadata.get("experiment") != "R2_DINO_original_SPVC_front_end_only_v1"
            or metadata.get("test_used") is not False
            or metadata.get("representation_sha256") != sha256(representation)
            or metadata.get("controller_sha256") != sha256(base_controller)
            or metadata.get("saved_policy_sha256") != sha256(saved_path)
            or metadata.get("saved_barrier_sha256") != sha256(barrier_path)):
        raise ValueError("step22 model hashes/provenance do not match; do not evaluate another PPO")
    scope_path = spvc_dir / "scope_comparison.json"
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    if any(scope.get(key) != metadata[key] for key in (
        "saved_policy_sha256", "saved_barrier_sha256"
    )):
        raise ValueError("scope comparison and saved checkpoint hashes disagree")
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
        raise ValueError("step22 changed the supposedly frozen q representation")
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
            "scope_comparison_sha256": sha256(scope_path),
            "changed_actor_tensors": changed,
            "q_parameters_unchanged": True,
            "state_distance_scale_m": adapted.state_distance_scale_m,
            "representation_distance_scale_m": adapted.distance_scale_m,
            "inference": "deterministic actor mean clipped to [-3,3]; no QP; no SBC action filtering",
        }


def check_adapter_equivalence(old, new, old_adapter, image_obs, q_obs, distance, speed):
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
    with torch.no_grad():
        vt = new.policy(torch.zeros((len(states), 4)), torch.from_numpy(states)).numpy()
    errors["new_vt_q_vs_physical_q_max_abs"] = float(np.max(np.abs(
        np.clip(vt, -3, 3)-new.predict(q_obs)[0]
    )))
    if not all(np.isfinite(value) and value <= 1e-5 for value in errors.values()):
        raise ValueError("actor loading/action equivalence check failed: %s" % errors)
    return errors
