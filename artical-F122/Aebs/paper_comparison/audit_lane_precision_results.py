"""Recompute CP ranks, physical transitions, actors and metrics from raw arrays."""
import argparse
import json
from pathlib import Path
import numpy as np

from .lane_precision import digest, precision_metrics, calibrate_performance, save


def main():
    import torch
    from Aebs.lane_keeping import core
    from Aebs.lane_keeping.cp import conformal_radius
    from Aebs.lane_keeping.policy import load_actor
    from Aebs.lane_keeping.robust_trajectory_cp import initial_draws
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera', type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads((args.camera/'protocol.json').read_text())
    protocol_hash = digest(args.camera/'protocol.json')
    records = {}
    for arm in protocol['arms']:
        out = args.camera/arm
        for filename, expected in protocol['bindings'][arm].items():
            assert digest(out/'package'/filename) == expected, (arm, filename)
        actor = load_actor(out/'package/actor.pt').requires_grad_(False)
        rows = {}
        for split, n_key, seed_key in [('calibration', 'calibration_n', 'calibration_seed'),
                                      ('holdout', 'test_n', 'test_seed')]:
            data = dict(np.load(out/(split+'.npz'), allow_pickle=False))
            initial, kappas, ids = initial_draws(protocol[seed_key], protocol[n_key])
            assert str(data['protocol_sha256']) == protocol_hash
            np.testing.assert_array_equal(data['states'][:, 0], initial)
            np.testing.assert_array_equal(data['kappas'], kappas)
            np.testing.assert_array_equal(data['episode_seeds'], ids)
            expected = core.dynamics_batch(data['states'][:, :-1].reshape(-1, 2),
                data['actions'].reshape(-1)*.45, np.repeat(kappas, 80)).reshape(len(initial), 80, 2)
            np.testing.assert_allclose(expected, data['states'][:, 1:], atol=1e-13, rtol=0)
            error = data['observations']-data['states'][:, :-1]/[.85, .30]
            np.testing.assert_allclose(error, data['errors'], atol=1e-13, rtol=0)
            np.testing.assert_allclose(abs(error).max(axis=(1, 2)), data['scores'], atol=1e-13, rtol=0)
            actor_error = 0.
            for first in range(0, len(initial), protocol['batch']):
                last = min(first+protocol['batch'], len(initial))
                for t in range(80):
                    inp = np.c_[data['observations'][first:last, t], kappas[first:last]/.01]
                    if len(inp) < protocol['batch']:
                        inp = np.r_[inp, np.repeat(inp[-1:], protocol['batch']-len(inp), axis=0)]
                    with torch.no_grad():
                        act = actor(torch.tensor(inp, dtype=torch.float32)).numpy().flatten()[:last-first]
                    actor_error = max(actor_error, float(abs(act-data['actions'][first:last, t]).max()))
            assert actor_error <= 1e-6, (arm, split, actor_error)
            metric = precision_metrics(data['states'], data['actions'])
            np.testing.assert_array_equal(metric['unsafe'], data['unsafe'])
            if split == 'calibration':
                q, rank = conformal_radius(data['scores'], .05)
                archived = json.loads((out/'calibration.json').read_text())
                assert q == archived['radius'] and rank == archived['rank']
                cp = calibrate_performance(metric)
                assert cp == json.loads((out/'precision_calibration.json').read_text())
            else:
                saved = np.load(out/'precision_metrics.npz')
                for key, values in metric.items():
                    np.testing.assert_array_equal(values, saved[key])
            rows[split] = dict(n=len(initial), raw_sha256=digest(out/(split+'.npz')),
                 maximum_actor_replay_error=actor_error, dynamics_checked=True,
                 residuals_checked=True, scenarios_reconstructed=True)
        records[arm] = rows
    save(args.camera/'independent_audit.json', dict(passed=True, protocol_sha256=protocol_hash,
         source_sha256=digest(__file__), per_arm=records,
         scope='implementation and data consistency; does not upgrade numerical barriers to exact proofs'))
    print(json.dumps(dict(passed=True, arms=len(records))), flush=True)


if __name__ == '__main__':
    main()
