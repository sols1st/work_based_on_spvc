"""Full SPVC controller pairs and RGB-DINO release on common camera replay.

Main rows pair each method's own front end and synthesized controller; crossed
rows attribute changes. Native SBC files remain attached to their own policies.
This evaluates deployed weights, not a matched-training-budget method ranking.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from Aebs.paper_comparison.aebs_intuitive_evaluate import (
    SCALE, FULL, PACKAGE, PPO_PATH, H5_PATH, load_actors, load_banks,
    predictions, collect, summary, sha, write)
from Combined_network.model import CombinedPolicyNetwork


NATIVE = Path('results/aebs_original_random_sbc_20261117')
CANDIDATES = dict(native_seed171='alternating',
                  native_seed172='seed202611172_alternating',
                  native_seed173='seed202611173_alternating')


class NativeActor(nn.Module):
    def __init__(self, path, original):
        super().__init__()
        self.register_buffer('C', torch.ones(1))
        import copy
        policy = copy.deepcopy(original.agent.policy)
        self.net = CombinedPolicyNetwork(policy.mlp_extractor.policy_net, policy.action_net)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        for prefix, filename in [('gen_net.', 'Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth'),
                                  ('state_net.', 'Aebs/controller/state_net_trained.pth')]:
            expected = torch.load(filename, map_location='cpu', weights_only=False)
            found = {k[len(prefix):]:v for k,v in checkpoint['policy'].items() if k.startswith(prefix)}
            if found.keys() != expected.keys() or any(not torch.equal(found[k], v) for k,v in expected.items()):
                raise ValueError('native front differs from replay prediction cache')
        weights = {k[len('controller_net.'):]: v for k, v in checkpoint['policy'].items()
                   if k.startswith('controller_net.')}
        self.net.load_state_dict(weights, strict=True)
        self.original_scale = original.original_scale

    def forward_observation(self, y, v):
        x = torch.cat([y * (SCALE/self.original_scale), v], 1)
        return self.net(x).clamp(-3, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=int, default=499)
    p.add_argument('--test', type=int, default=2000)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    actors = load_actors()
    native_records = {}
    inputs = [Path(__file__), Path('Aebs/paper_comparison/aebs_intuitive_evaluate.py'),
              Path('Aebs/vision_sls/rollout.py'), Path('Aebs/system/outcomes.py'),
              PPO_PATH, H5_PATH, FULL/'estimates.npz',
              PACKAGE/'policy.pt', PACKAGE/'observation.pt', PACKAGE/'sbc.pt']
    for key, folder in CANDIDATES.items():
        path = NATIVE/folder/'best.pt'
        record = json.loads((path.parent/'summary.json').read_text())
        if record['weights']['best.pt'] != sha(path):
            raise ValueError('native policy/barrier checksum')
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        if not ckpt.get('barrier') or ckpt['iteration'] != record['best']['iteration']:
            raise ValueError('missing native matching barrier')
        actors[key] = NativeActor(path, actors['original_ppo']).eval()
        native_records[key] = dict(pair_sha256=sha(path), iteration=ckpt['iteration'],
            original_training_score=record['best']['score'],
            source_summary=str(path.parent/'summary.json'),
            certificate='source acceptance only; independent drift audit failed; not a camera-law guarantee')
        inputs += [path, path.parent/'summary.json', path.parent/'protocol.json']
    banks = load_banks()
    protocol = dict(version='native_spvc_full_weights_common_replay_v1',
        inputs={str(f): sha(f) for f in inputs}, native_pairs=native_records,
        calibration_n=args.calibration, test_n=args.test, alpha=.05,
        seeds=dict(calibration=202611230201, test=202611230202),
        initial='IID uniform [15,16]m x [2.5,3]m/s; same initial and appearance per arm',
        cameras={k:len(v['filename']) for k,v in banks.items()},
        primary_bank='common:3229 image IDs excluding training/development images of registered budgets',
        secondary_bank='full24k:24000 images including fitting data, deployment-bank sensitivity only',
        actor_keys=list(actors), fronts=['native_m','dino_m'],
        primary_pairs='native_seed17{1,2,3}+native_m versus current_robust+dino_m',
        other_pairs='crossed front/controller attribution; original_ppo is initialization diagnostic',
        own_rgb='640x640 real RGB sources -> RGB DINO preprocessing; cached frozen observations',
        original_gray='original cGAN and state estimator weights; actual camera gray32x32 -> own controller',
        physics='d_next=d-.05v; v_next=clip(v-.05a,0,3), a in [-3,3]m/s^2',
        success='d in [5,6]m and v<=.5m/s; danger=d<=6m and v>.5; max400 steps',
        cp='separate post-freeze calibration for each closed-loop arm; max |dhat-d|; physical-distance diagnostic for native (not native SBC noise law)',
        training='existing method-specific histories; original SPVC 400-H5 initialization/native alternating synthesis versus our 24k RGB and robust synthesis; not equal training budgets',
        guarantee='own numerical barrier only on its registered <=.45m tube; new q>.45m cannot inherit it; native source SBC does not transfer to real-camera law',
        scope='finite frozen-bank replay; new seeds are new scenarios, not new image collection',
        inference='paired episode bootstrap4000; conditional on frozen checkpoints/banks; three native seeds not pooled as6000 independent scenarios')
    write(out/'protocol.json', protocol)
    for name, bank in banks.items():
        np.savez_compressed(out/(name+'_image_ids.npz'), **bank)
    threshold = {}
    rows = {}
    records = {}
    start = time.time()
    with torch.inference_mode():
        for phase, n in [('calibration', args.calibration), ('test', args.test)]:
            rng = np.random.default_rng(protocol['seeds'][phase])
            initial = rng.uniform([15.,2.5], [16.,3.], (n,2))
            appearance = rng.integers(0,12,size=n)
            for bank_name, bank in banks.items():
                for field in protocol['fronts']:
                    model = dict(prediction_path=str(FULL/'estimates.npz'), field=field,
                                 ids_field='files', prediction_units='meters')
                    pred = predictions(model, bank)
                    for actor_name, actor in actors.items():
                        key = bank_name+'__'+field+'__'+actor_name
                        r = collect(actor, bank, pred, initial, appearance)
                        np.savez_compressed(out/(key+'__'+phase+'.npz'), **r)
                        if phase == 'calibration':
                            rank = math.ceil((n+1)*.95)
                            threshold[key] = float(np.sort(r['scores'])[rank-1]) if rank<=n else float('inf')
                        else:
                            rows[key] = summary(r, threshold[key])
                            records[key] = r
                        print(json.dumps(dict(phase=phase, arm=key,
                            success=int((r['outcomes']=='success').sum()), q_m=threshold[key])), flush=True)
            if phase == 'calibration':
                write(out/'frozen_thresholds.json', threshold)
        paired = {}
        bootstrap = np.random.default_rng(202611230203)
        for bank_name in banks:
            current = records[bank_name+'__dino_m__current_robust']
            for native in CANDIDATES:
                old = records[bank_name+'__native_m__'+native]
                np.testing.assert_array_equal(current['initial'], old['initial'])
                np.testing.assert_array_equal(current['appearance'], old['appearance'])
                delta = (current['outcomes']=='success').astype(float)-(old['outcomes']=='success')
                boot = bootstrap.choice(delta, (4000,len(delta))).mean(1)
                paired[bank_name+'__'+native] = dict(success_improvement_fraction=float(delta.mean()),
                    bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    current_only=int((delta==1).sum()), native_only=int((delta==-1).sum()))
    if any(sha(f)!=protocol['inputs'][str(f)] for f in inputs):
        raise ValueError('input changed during execution')
    write(out/'summary.json', dict(arms=rows, paired=paired, elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    main()
