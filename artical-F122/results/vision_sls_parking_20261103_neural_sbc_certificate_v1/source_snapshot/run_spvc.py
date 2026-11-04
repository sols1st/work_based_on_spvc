"""3D VISION-SLS front end + original SPVC learner first connection experiment.

Keeps original backend files unchanged. This is sampled training/diagnosis,
not a completed continuous-state verifier or infinite-horizon certificate.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppo-dir', default='results/vision_sls_parking_20261025_ppo_v2')
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--barrier-epochs', type=int, default=100)
    p.add_argument('--actor-epochs', type=int, default=1)
    p.add_argument('--actor-lr', type=float, default=.05,
                   help='Original SPVC default .05; explicit scenario adaptation only')
    p.add_argument('--image-eval', action='store_true')
    p.add_argument('--vision-root', default='../VISION-SLS')
    p.add_argument('--dino-repo', default='../external/dinov2')
    p.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    a = p.parse_args()
    out, source = Path(a.output_dir), Path(a.ppo_dir)
    if out.exists() or min(a.barrier_epochs, a.actor_epochs) < 1 or not np.isfinite(a.actor_lr) or a.actor_lr <= 0:
        p.error('new output directory and positive epoch counts required')
    from .train_ppo import sha, evaluate
    from .ppo_env import FrozenImageSensor
    from .spvc_backend import (ParkingSPVCPolicy, ParkingSPVCSystem, collect_replay,
                               make_original_learner, sampled_barrier_diagnostic)
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    torch.manual_seed(7)
    np.random.seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_metrics = json.loads((source/'metrics.json').read_text())
    source_config = json.loads((source/'config.json').read_text())
    if (source_metrics['checkpoint_sha256'] != sha(source/'ppo.zip') or
            source_config['projection_sha256'] != sha(a.projection)):
        p.error('base actor/projection provenance mismatch')
    backend = [Path('Aebs/VT')/n for n in ('train.py', 'verify.py', 'loop.py', 'utils.py')]
    hashes = {str(f): sha(f) for f in backend}
    out.mkdir(parents=True)
    cfg = dict(vars(a), front_end_dimension=3, actor_input_dimension=16,
               augmented_barrier_state_dimension=20,
               backend='original Aebs.VT.train.VTLearner and VTVerifier.compute_expected_l',
               backend_sha256=hashes, projection_sha256=sha(a.projection),
               source_ppo_sha256=sha(source/'ppo.zip'), seed=7,
               noise='zero, same as previous parking test; no image error distribution added',
               verification='point diagnostic only; full 20D grid NOT run',
               region_sampler='original parking circles/bounds; initial exact 15 starts',
               source_directory=str(source))
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    started = time.monotonic()
    base = PPO.load(source/'ppo.zip', device='cpu')
    initial, states = collect_replay(base)
    system = ParkingSPVCSystem(initial, states)
    policy = ParkingSPVCPolicy(base)
    learner = make_original_learner(system, policy)
    for group in learner.p_optimizer.param_groups:
        group['lr'] = a.actor_lr
    actor_before = {k: v.detach().clone() for k, v in learner.p_net.state_dict().items()}
    barrier_before = {k: v.detach().clone() for k, v in learner.l_model.state_dict().items()}
    before = sampled_barrier_diagnostic(learner, system, states)
    np.savez_compressed(out/'training_replay.npz', initial=initial, augmented_states=states)
    print('[3D front end -> original SPVC backend connection]', flush=True)
    print('replay=%d augmented_dim=20, actor_input=16, visual_dim=3' % len(states), flush=True)
    # Call the ORIGINAL training methods; this pilot uses recorded states with
    # zero cell-jitter instead of pretending a 20D full-state grid was verified.
    l_history = learner.train_epoch(states, np.zeros(20), lip=.001, batch_size=256,
                                    num_epochs=a.barrier_epochs, train_fn='l')
    print('barrier final:', l_history[-1], flush=True)
    after_l = sampled_barrier_diagnostic(learner, system, states)
    actor_unchanged_after_l = all(torch.equal(v, actor_before[k])
                                  for k, v in learner.p_net.state_dict().items())
    barrier_trained = {k: v.detach().clone() for k, v in learner.l_model.state_dict().items()}
    p_history = learner.train_epoch(states, np.zeros(20), lip=.001, batch_size=256,
                                    num_epochs=a.actor_epochs, train_fn='p')
    print('actor final:', p_history[-1], flush=True)
    after_p = sampled_barrier_diagnostic(learner, system, states)
    torch.save(learner.l_model.state_dict(), out/'sbc.pt')
    torch.save(learner.p_net.state_dict(), out/'policy_spvc.pt')
    candidate = learner.p_net.export_actor(copy.deepcopy(base))
    candidate.save(out/'ppo.zip')
    # Verify exact interpretation of the exported PPO, including action clipping.
    with torch.no_grad():
        expected = learner.p_net(torch.zeros(len(states), 4, device=learner.device),
                                 torch.as_tensor(states, device=learner.device)).cpu().numpy()
    from .core import ACTION_HIGH
    exported, _ = candidate.predict(states[:, 4:], deterministic=True)
    export_gap = float(np.max(np.abs(exported*ACTION_HIGH-expected)))
    if export_gap > 1e-4:
        raise ValueError('exported PPO/SPVC actor action mismatch: %.9g' % export_gap)
    def changed(old, net):
        return sum(not torch.equal(v, old[k]) for k, v in net.state_dict().items())
    metrics = dict(config=cfg, before_training=before, after_barrier_training=after_l,
                   after_actor_training=after_p, l_history=l_history, p_history=p_history,
                   actor_changed_tensors=changed(actor_before, learner.p_net),
                   barrier_changed_tensors=changed(barrier_before, learner.l_model),
                   actor_unchanged_during_barrier_phase=actor_unchanged_after_l,
                   barrier_unchanged_during_actor_phase=all(torch.equal(v, barrier_trained[k])
                       for k, v in learner.l_model.state_dict().items()),
                   actor_export_max_abs_physical_action_error=export_gap,
                   baseline={k: source_metrics[k] for k in ('proxy', 'image')},
                   baseline_image_result_reused_with_matching_checkpoint=True,
                   checkpoint_sha256=sha(out/'ppo.zip'), training_seconds=time.monotonic()-started,
                   full_verifier_run=False, probability_guarantee=None)
    paths = {}
    metrics['proxy'], paths['proxy_states'], paths['proxy_actions'] = evaluate(candidate)
    candidate_starts, candidate_states = collect_replay(candidate)
    metrics['candidate_own_rollout_barrier'] = sampled_barrier_diagnostic(learner, system, candidate_states)
    np.savez_compressed(out/'candidate_replay.npz', initial=candidate_starts, augmented_states=candidate_states)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz', **paths)
    if a.image_eval:
        sensor = FrozenImageSensor(a.vision_root, a.dino_repo, a.dino_weights, a.projection)
        try:
            metrics['image'], paths['image_states'], paths['image_actions'] = evaluate(candidate, sensor)
        finally:
            sensor.close()
    # Preserve base controller irrespective of outcome; a passing simulator run
    # would still NOT qualify as a completed SBC certificate.
    metrics['candidate_promoted'] = False
    metrics['status'] = 'spvc_backend_connection_complete_not_certified'
    metrics['backend_files_unchanged'] = hashes == {str(f): sha(f) for f in backend}
    metrics['projection_unchanged'] = cfg['projection_sha256'] == sha(a.projection)
    metrics['source_ppo_unchanged'] = cfg['source_ppo_sha256'] == sha(source/'ppo.zip')
    metrics['checkpoint_unchanged_during_evaluation'] = metrics['checkpoint_sha256'] == sha(out/'ppo.zip')
    metrics['runtime_seconds'] = time.monotonic()-started
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    np.savez_compressed(out/'rollouts.npz', **paths)
    for path in ('proxy', 'image'):
        if path in metrics:
            old, new = metrics['baseline'][path], metrics[path]
            print('%s success %d -> %d /15; collision %d -> %d; bounds %d -> %d' %
                  (path, old['success_count'], new['success_count'], old['collision_count'],
                   new['collision_count'], old['out_of_bounds_count'], new['out_of_bounds_count']), flush=True)
    print('metrics:', out/'metrics.json', flush=True)


if __name__ == '__main__':
    main()
