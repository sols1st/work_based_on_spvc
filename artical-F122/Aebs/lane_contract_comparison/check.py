"""Recompute stored CP ranks, nominal rollouts, and feasible witnesses from the portable result."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..lane_keeping import core
from ..lane_keeping.learned_sbc import load_sbc_checkpoint
from ..lane_keeping.policy import load_actor


def check_hash(path, expected):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path


def evaluate(actor, positions, curvatures):
    curvature = np.broadcast_to(curvatures, positions.shape[:-1]) / core.KAPPA_SCALE
    inputs = np.concatenate((positions, curvature[..., None]), axis=-1)
    flat = torch.tensor(inputs, dtype=torch.float64).reshape(-1, 3)
    return actor(flat).flatten().numpy().reshape(inputs.shape[:-1])


@torch.inference_mode()
def main(root):
    torch.set_num_threads(2)
    load = lambda filename: json.loads((root / filename).read_text())
    protocol, contract, cov = map(load, ("protocol.json", "contracts.json", "coverage.json"))
    check_hash(root / "protocol.json", contract["protocol"])
    check_hash(root / "contracts.json", cov["contracts_sha256"])
    check_hash(root / "calibration.npz", contract["calibration"])
    check_hash(root / "holdout.npz", cov["holdout_sha256"])
    check_hash(root / "inputs/development.npz", protocol["development"]["sha256"])
    for name, key in (("actor", "actor"), ("sbc", "sbc"), ("patch_projection", "observer")):
        check_hash(root / f"inputs/package/{name}.pt", protocol["binding"][f"{key}_sha256"])
    actor = load_actor(root / "inputs/package/actor.pt").double().eval()
    barrier = load_sbc_checkpoint(root / "inputs/package/sbc.pt")[0].double().eval()
    xscale = np.array([core.E_SCALE, core.THETA_SCALE])
    scale = protocol["development"]["scale"]
    vscale, ascale = np.array(scale["visual"]), scale["action"]
    for split, filename in (("development", "inputs/development.npz"),
                            ("calibration", "calibration.npz"), ("holdout", "holdout.npz")):
        with np.load(root / filename, allow_pickle=False) as data:
            state, obs = data["states"], data["observations"]
            assert state.shape == (protocol["holdout_episodes"] if split == "holdout" else 199,
                                   protocol["horizon"] + 1, 2)
            k = data["kappas"][:, None]
            normalized = state[:, :-1] / xscale
            obs_error = np.abs(obs - normalized)
            np.testing.assert_allclose(obs_error, np.abs(data["errors"]), atol=1e-14)
            seen = evaluate(actor, obs, k)
            np.testing.assert_allclose(seen, data["actions"], atol=2e-6)
            act_error = np.abs(seen - evaluate(actor, normalized, k))
            if split == "development":
                np.testing.assert_allclose(np.quantile(obs_error.max(1), .95, axis=0), vscale)
                np.testing.assert_allclose(np.quantile(act_error.max(1), .95), ascale)
                continue
            k_all = np.broadcast_to(k, seen.shape).flatten()
            nxt = core.dynamics_batch(state[:, :-1].reshape(-1, 2),
                                      seen.flatten() * core.PHI_MAX, k_all)
            np.testing.assert_allclose(nxt.reshape(state[:, 1:].shape), state[:, 1:], atol=2e-12)
            diag = (obs_error / vscale).max(2)
            scores = {"uniform": obs_error.max((1, 2)),
                      "diagonal": diag.max(1),
                      "joint": np.maximum(.5 * diag, act_error / ascale).max(1)}
            for name, values in scores.items():
                arm = contract["arms"][name]
                if split == "calibration":
                    rank = int(np.ceil((len(values) + 1) * (1 - protocol["alpha"])))
                    assert rank == arm["rank"]
                    np.testing.assert_allclose(np.sort(values)[rank - 1], arm["q"], atol=2e-11)
                    continue
                mask = values <= arm["q"]
                inside = (obs_error <= np.array(arm["width"])).all(axis=(1, 2))
                if arm["action"] is not None:
                    inside &= (act_error <= arm["action"]).all(axis=1)
                np.testing.assert_array_equal(inside, mask)
                assert int(mask.sum()) == cov["arms"][name]["covered"]
            if split == "holdout":
                assert (np.abs(state / xscale) > 1).any((1, 2)).sum() == cov["unsafe"]
                np.testing.assert_allclose(np.abs(state[:, 1:, 0]).mean(), cov["mean_abs_e_m"])

    def get_commands(witness, arm, key):
        x = np.array(witness[key])
        e = np.array(witness["error"])
        assert (np.abs(e) <= np.array(arm["width"]) + 1e-12).all()
        obs = float(evaluate(actor, x + e, witness["kappa"]))
        nom = float(evaluate(actor, x, witness["kappa"]))
        if arm["action"] is not None:
            assert abs(obs - nom) <= arm["action"] + 1e-12
        return x, obs, nom

    ranges = load("action_range.json")
    check_hash(root / "contracts.json", ranges["contracts"])
    for name, row in ranges["arms"].items():
        x, obs, nom = get_commands(row["witness"], contract["arms"][name], "state")
        assert (np.abs(x * xscale) <= [core.INIT_E, core.INIT_THETA]).all()
        np.testing.assert_allclose(abs(obs - nom), row["attained_lower"])
        assert row["continuous_upper"] >= abs(obs - nom) - 1e-12

    verify, refine = load("verification.json"), load("refinement.json")
    check_hash(root / "verification.json", refine["coarse"])
    for name, arm in contract["arms"].items():
        for group in (verify["runs"][name], refine["arms"][name]["runs"]):
            for margin, result in group.items():
                if result["status"] != "witness":
                    continue
                point = next(row["witness"] for row in result["curvatures"].values()
                             if row["status"] == "witness")
                x, obs, nom = get_commands(point, arm, "x")
                delta = point["additive_action"]
                assert abs(delta) <= float(margin) and (np.abs(x) <= 1).all()
                nxt = core.dynamics(x * xscale, (obs + delta) * core.PHI_MAX, point["kappa"]) / xscale
                np.testing.assert_allclose(nxt, point["next_state"], atol=1e-11)
                b0, b1 = barrier(torch.tensor(np.stack((x, nxt)))).flatten().numpy()
                np.testing.assert_allclose([b0, b1], [point["before"], point["after"]], atol=1e-10)
                assert b1 > b0 and b0 <= 20
    print("Verified saved model hashes, calibration ranks, full holdout masks, transitions, "
          "steering examples and barrier witnesses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
