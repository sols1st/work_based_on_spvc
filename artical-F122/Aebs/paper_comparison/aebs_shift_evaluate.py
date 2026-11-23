"""All preregistered visual shifts on frozen AEBS models, without new CP calibration."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from Aebs.paper_comparison.aebs_intuitive_evaluate import (
    ACTORS, TEST_N, TEST_SEED, collect, design, episodes, load_actors, load_banks,
    predictions, read, sha, static_metrics, summary, write,
)


def protocol(manifest, nominal):
    token = design(manifest)
    manifest_record = read(manifest)
    if 'source_manifest' in manifest_record:
        source_manifest = Path(manifest_record['source_manifest'])
        if sha(source_manifest) != manifest_record['source_manifest_sha256']:
            raise ValueError('changed upstream complete prediction manifest')
        token['inputs'][str(source_manifest)] = sha(source_manifest)
        token['subset_rule'] = manifest_record['subset_rule']
    if len(token['models']) != 54 or len(token['arms']) != 108:
        raise ValueError('expected54 fronts and108 actor arms')
    conditions = sorted(set(m['condition'] for m in token['models']))
    if len(conditions) != 6 or any(m['budget'] != 24000 for m in token['models']):
        raise ValueError('expected six shifts at fixed largest24000 budget')
    modes = sorted(set(m['mode'] for m in token['models']))
    seeds = sorted(set(m['train_seed'] for m in token['models']))
    if len(modes) != 3 or len(seeds) != 3:
        raise ValueError('expected all3 methods and3 training seeds')
    tuples = {(m['condition'], m['mode'], m['train_seed']) for m in token['models']}
    if len(tuples) != 54:
        raise ValueError('unbalanced shift design')
    token.update(version='aebs_frozen_visual_shift_v1', nominal_output=str(nominal),
                 n={'holdout': TEST_N}, seeds={'holdout': TEST_SEED}, alpha=None,
                 conditions=conditions, formal_scope='test-only preregistered visual stress; no new CP guarantee',
                 cp='Report empirical fraction inside frozen nominal q only; shifted law violates nominal calibration assumptions, so no95% coverage claim.',
                 calibration='none; models and nominal thresholds frozen before shift evaluation')
    token['inputs'][str(Path(__file__))] = sha(Path(__file__))
    token['inputs'][str(nominal / 'protocol.json')] = sha(nominal / 'protocol.json')
    token['inputs'][str(nominal / 'summary.json')] = sha(nominal / 'summary.json')
    return token


def run(args):
    torch.set_num_threads(1)
    token = protocol(args.manifest, args.nominal)
    if args.register:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / 'protocol.json', token)
        print('registered108 shifts', flush=True)
        return
    if read(args.output / 'protocol.json') != token or (args.output / 'summary.json').exists():
        raise ValueError('protocol differs or results already exist')
    nominal = read(args.nominal / 'summary.json')['arms']
    initial, appearance = episodes(TEST_N, TEST_SEED)
    banks, actors = load_banks(), load_actors()
    bank = banks['common']
    results, static = {}, {}
    np.savez_compressed(args.output / 'holdout_scenarios.npz', initial=initial, appearance=appearance)
    start = time.perf_counter()
    for arm in token['arms']:
        m = token['models'][arm['model']]
        matches = [k for k, v in nominal.items() if
                   v['model']['mode'] == m['mode'] and v['model']['budget'] == m['budget'] and
                   v['model']['train_seed'] == m['train_seed'] and v['arm']['actor'] == arm['actor']]
        if len(matches) != 1:
            raise ValueError('missing unique nominal pairing')
        nominal_key = matches[0]
        q = nominal[nominal_key]['q_m']
        pred = predictions(m, bank)
        static[m['key']] = static_metrics(bank, pred)
        one_start = time.perf_counter()
        data = collect(actors[arm['actor']], bank, pred, initial, appearance)
        dest = args.output / arm['key']
        dest.mkdir(exist_ok=True)
        np.savez_compressed(dest / 'holdout.npz', **data)
        results[arm['key']] = dict(summary(data, q), arm=arm, model=m, nominal_key=nominal_key,
                                  q_scope='frozen nominal threshold only, no shifted CP guarantee',
                                  seconds=time.perf_counter() - one_start,
                                  result_sha256=sha(dest / 'holdout.npz'))
        print(arm['key'], results[arm['key']]['outcomes'], flush=True)
    if protocol(args.manifest, args.nominal) != token:
        raise ValueError('inputs changed')
    write(args.output / 'static_metrics.json', static)
    write(args.output / 'summary.json', dict(arms=results, total_seconds=time.perf_counter() - start,
          protocol_sha256=sha(args.output / 'protocol.json'), formal=True, calibrated=False,
          scope='conditional on fixed camera library, shift condition, and frozen models; no shifted-law CP guarantee'))
    print('complete', time.perf_counter() - start, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest', type=Path)
    p.add_argument('nominal', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--register', action='store_true')
    run(p.parse_args())
