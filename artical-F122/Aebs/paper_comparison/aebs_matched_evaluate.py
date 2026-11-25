"""Evaluate each complete AEBS method with its own frozen frontend and controller."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.conformal import cp_quantile
from Aebs.vision_sls.model import VisionSLSPolicy, load_projection
from Combined_network.model import AebsEnd2EndNet, SubNet
from cGAN.taxi_models_and_data import AebsMLPGenerator
from .aebs_intuitive_evaluate import collect, static_metrics, summary
from .aebs_matched_data import write_json
from .aebs_matched_native import sha
from .aebs_matched_rgb import CONDITIONS, batches, extract, transform_image
from .aebs_matched_robust import contract, fresh_policy, library, run_stage, scenario

CAL_SEED = 2026112503
TEST_SEED = 2026112504


def read(path):
    return json.loads(path.read_text())


def freeze(experiment, output, count_cal, count_test):
    own_run = read(experiment / 'own_synthesis/synthesis.json')
    native_run = read(experiment / 'synthesis.json')
    native_root = experiment / 'upstream'
    own_package = Path(own_run['package'])
    native_pair = experiment / native_run['selected']
    ppo = native_root / 'Aebs/controller/best_model/best_model.zip'
    native_front = native_root / 'Aebs/controller/state_net_trained.pth'
    generator = native_root / 'Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth'
    assert sha(native_pair) == native_run['weights_sha256']
    assert all(sha(own_package / name) == digest for name, digest in own_run['weights'].items())
    assert sha(experiment / 'own_initial/observation.pt') == sha(own_package / 'observation.pt')
    output.mkdir(exist_ok=False)
    tested_files = [own_package / n for n in ('policy.pt', 'observation.pt', 'sbc.pt', 'config.json')]
    tested_files.extend([native_pair, ppo, native_front, generator, experiment / 'manifest.npz', Path(__file__)])
    write_json(output / 'evaluation_protocol.json', dict(
        sources={str(f): sha(f) for f in tested_files}, clean_train_only=True,
        conditions=dict(CONDITIONS), calibration_n=count_cal, test_n=count_test,
        calibration_seed=CAL_SEED, test_seed=TEST_SEED, alpha=.05,
        primary='SPVC retrained generator+estimator+alternating actor versus RGB DINO+robust actor',
        training_stage_row='native paired grayscale state estimator + native retrained PPO before alternating synthesis',
        test='same frozen holdout camera bank in both arms; separately sampled calibration and test episodes',
        initial='uniform [15,16] m, [2.5,3] m/s, and one shared draw of appearance per trajectory',
        termination='400 steps; common first-entry success [5,6]m with speed<=.5; danger d<=6m and v>.5',
        clean_only_certificate=True))
    return own_package, native_pair, ppo, native_front, generator


def native_components(scale, ppo_path, pair, front, gen_path):
    ppo_actor = fresh_policy(ppo_path, scale)
    params = PPO.load(ppo_path, device='cpu').policy
    gen = AebsMLPGenerator(4, 1)
    net = AebsEnd2EndNet(gen, [1024, 256, 64, 1],
                        copy.deepcopy(params.mlp_extractor.policy_net), copy.deepcopy(params.action_net))
    checkpoint = torch.load(pair, map_location='cpu', weights_only=False)
    net.load_state_dict(checkpoint['policy'])
    front_weights = torch.load(front, map_location='cpu', weights_only=True)
    gen_weights = torch.load(gen_path, map_location='cpu', weights_only=True)
    for module, saved in [(net.state_net, front_weights), (net.gen_net, gen_weights)]:
        assert all(torch.equal(tensor, saved[name]) for name, tensor in module.state_dict().items())
    net.eval().requires_grad_(False)
    native_actor = VisionSLSPolicy(net.controller_net, scale).eval().requires_grad_(False)
    with torch.inference_mode():
        rng = np.random.default_rng(2026112505)
        states = torch.tensor(rng.uniform([5 / scale, 0], [16 / scale, 3], (256, 2)), dtype=torch.float32)
        z = torch.tensor(rng.uniform(-1, 1, (256, 4)), dtype=torch.float32)
        expected = net(z, states).clamp(-3, 3)
        obs = net.state_net(net.gen_net(z, states[:, :1]).flatten(1))
        actual = native_actor.forward_observation(obs, states[:, 1:2])
        pair_difference = float((expected - actual).abs().max())
        assert pair_difference <= 1e-7
    return native_actor, ppo_actor, net.state_net, pair_difference


def cache_features(experiment, device, workers):
    target = experiment / 'holdout_features'
    target.mkdir(exist_ok=True)
    data = dict(np.load(experiment / 'manifest.npz'))
    ids = data['files'][data['split'] == 'holdout']
    source = Path(read(experiment / 'data_protocol.json')['source'])
    projection = torch.load(experiment / 'own_initial/observation.pt', map_location='cpu', weights_only=False)
    assert sha(Path(projection['dino_weights'])) == projection['provenance']['dino_weights_sha256']
    backbone = FrozenDinoV2(projection['dino_repo'], projection['dino_weights'], device)
    for name, _ in CONDITIONS:
        dest = target / (name + '.pt')
        if not dest.exists():
            features = extract(backbone, source, ids, name, device, workers)
            torch.save(dict(features=features, ids=ids.tolist(),
                            weights_sha256=projection['provenance']['dino_weights_sha256']), dest)
        print('cached', name, flush=True)


def native_predictions(model, source, files, condition, device, workers):
    # The DINO reader has done its RGB inference. Read gray input with the
    # identical pre-processing implementation, without transferring RGB to CUDA.
    model = model.to(device)
    parts = []
    with torch.inference_mode():
        for _, gray in batches(source, files, condition, workers, include_gray=True):
            parts.append(model(gray.to(device, non_blocking=True).float() * (2 / 255.) - 1).flatten().cpu())
    return torch.cat(parts).numpy()


def snapshots(source, bank, output):
    picks = []
    for group in (0, 5):
        ids = np.flatnonzero(bank['appearance'] == group)
        picks.append(ids[np.argmin(abs(bank['distance_m'][ids] - (8 if group == 0 else 14)))])
    width = 200
    board = Image.new('RGB', (width * len(CONDITIONS), (width + 26) * len(picks)), '#ffffff')
    draw = ImageDraw.Draw(board)
    for y, index in enumerate(picks):
        with Image.open(source / bank['files'][index]) as raw:
            raw = raw.convert('RGB')
            for x, (name, _) in enumerate(CONDITIONS):
                im = transform_image(raw, name).resize((width, width), Image.Resampling.LANCZOS)
                board.paste(im, (x * width, y * (width + 26)))
                draw.text((x * width + 4, y * (width + 26) + width + 4),
                          f'{name}  {bank["distance_m"][index]:.2f} m', fill='black')
    board.save(output / 'input_examples.png')


def evaluate(experiment, count_cal, count_test, device, workers):
    output = experiment / 'evaluation'
    own_package, native_pair, ppo, front, gen = freeze(experiment, output, count_cal, count_test)
    metadata = read(experiment / 'data_protocol.json')
    scale = metadata['train_distance_scale_m']
    source = Path(metadata['source'])
    data = dict(np.load(experiment / 'manifest.npz'))
    bank = library(data, data['split'] == 'holdout')
    projection, project_weights = load_projection(own_package / 'observation.pt', device)
    base = fresh_policy(ppo, scale)
    own_actor = copy.deepcopy(base)
    own_actor.load_state_dict(torch.load(own_package / 'policy.pt', map_location='cpu', weights_only=False)['state_dict'])
    native_actor, ppo_actor, gray_model, max_gap = native_components(scale, ppo, native_pair, front, gen)
    actors = [('spvc', native_actor), ('spvc_ppo', ppo_actor), ('ours', own_actor)]
    scenarios = {'calibration': scenario(count_cal, CAL_SEED), 'test': scenario(count_test, TEST_SEED)}
    rows = {}
    for condition, title in CONDITIONS:
        features = torch.load(experiment / 'holdout_features' / (condition + '.pt'),
                              map_location='cpu', weights_only=True)
        assert features['ids'] == bank['files'].tolist()
        assert features['weights_sha256'] == project_weights['provenance']['dino_weights_sha256']
        with torch.inference_mode():
            rgb_pred = projection(features['features'].to(device)).flatten().cpu().numpy()
        gray_pred = native_predictions(gray_model, source, bank['files'], condition, device, workers)
        pred_m = dict(spvc=gray_pred * scale, ours=rgb_pred * scale)
        np.savez_compressed(output / (condition + '_static.npz'), **bank, **pred_m)
        row = dict(title=title, static={name: static_metrics(bank, v / scale, scale=scale) for name, v in pred_m.items()}, closed_loop={})
        q_values = {}
        for phase, (initial, appearance) in scenarios.items():
            for name, actor in actors:
                pred = pred_m['ours' if name == 'ours' else 'spvc'] / scale
                result = collect(actor, bank, pred, initial, appearance, scale=scale)
                np.savez_compressed(output / f'{condition}__{name}__{phase}.npz', **result)
                if phase == 'calibration':
                    q_values[name] = cp_quantile(result['scores'], .05)
                else:
                    row['closed_loop'][name] = summary(result, q_values[name]['q_hat'])
            if phase == 'calibration':
                write_json(output / (condition + '_frozen_cp.json'), q_values)
        rows[condition] = row
        write_json(output / 'results.json', rows)
        print('completed', condition, flush=True)
    snapshots(source, bank, output)
    clean = rows['clean']['closed_loop']['ours']
    clean_contract = output / 'clean_robust_contract.json'
    write_json(clean_contract, contract(own_package, clean['q_m'], independent_trajectory_calibration=True,
                                       statistical_status='marginal CP on frozen RGB holdout replay law'))
    audit = run_stage(output / 'clean_audit', 'Aebs.vision_sls.verify_sbc_cells',
                      '--artifact-dir', own_package, '--barrier-dir', own_package,
                      '--perception-contract', clean_contract, '--output-dir', output / 'clean_audit',
                      '--actor-bound-method', 'affine', '--stop-on-witness',
                      '--max-depth', 20, '--max-cells', 1000000)
    assert all(sha(Path(f)) == dig for f, dig in read(output / 'evaluation_protocol.json')['sources'].items())
    write_json(output / 'completed.json', dict(native_action_adapter_max_difference_mps2=max_gap,
                                              clean_robust_verification=audit,
                                              conditions=list(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['cache', 'evaluate'])
    parser.add_argument('experiment', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--calibration', type=int, default=499)
    parser.add_argument('--test', type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.mode == 'cache':
        cache_features(args.experiment, args.device, args.workers)
    else:
        evaluate(args.experiment, args.calibration, args.test, args.device, args.workers)


if __name__ == '__main__':
    main()
