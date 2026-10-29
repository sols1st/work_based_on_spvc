"""Train and execute the VISION-SLS front end with the original SPVC backend.

Run from artical-F122: python -m Aebs.vision_sls.run --smoke
No simulator is required for saved-image replay; it is not live CARLA.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3 import PPO

from Aebs.dino_latent.backbone import FrozenDinoV2
from Aebs.vision_sls.model import ObservationProjection, ImageObservation, VisionSLSPolicy
from Aebs.vision_sls.env import ObservationEnv
from Aebs.vision_sls.rollout import rollout
from Combined_network.model import CombinedPolicyNetwork

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'Aebs/data/Downsampled.h5')
    parser.add_argument('--dino-repo', type=Path, default=ROOT.parent/'external/dinov2')
    parser.add_argument('--dino-weights', type=Path, default=ROOT.parent/'external/dinov2_weights/dinov2_vits14_pretrain.pth')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/vision_sls_spvc')
    parser.add_argument('--feature-cache', type=Path, help='Reuse a matching features.pt from an earlier run')
    parser.add_argument('--smoke', action='store_true', help='Small PPO/VT budgets; not a certificate run')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--ppo-steps', type=int, default=None)
    parser.add_argument('--vt-seconds', type=float, default=3600.)
    parser.add_argument('--max-iteration-index', type=int, default=None)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cpu','cuda','mps'], default='cpu')
    parser.add_argument('--actor-learning-rate',type=float,default=3e-4,
                        help='SPVC actor step size; legacy 0.05 can destroy a working PPO')
    return parser


def train_observation(args, out):
    with h5py.File(args.data, 'r') as data:
        images = np.asarray(data['X_train'], dtype=np.float32)
        distances = np.asarray(data['y_train'], dtype=np.float32).reshape(-1)
    if len(images) != len(distances) or len(images) < 20 or not np.isfinite(images).all():
        raise ValueError('at least 20 paired finite images and distance labels required')
    if images.min() < 0 or images.max() > 1 or not np.isfinite(distances).all():
        raise ValueError('dataset must contain [0,1] images and finite distance labels')
    # Keep byte-identical images in one split, preventing duplicate-image leakage.
    groups = {}
    for i, image in enumerate(images):
        groups.setdefault(hashlib.sha256(image.tobytes()).hexdigest(), []).append(i)
    keys = list(groups)
    if len(keys) < 10:
        raise ValueError('at least 10 distinct images required for disjoint splits')
    order = np.random.default_rng(args.seed).permutation(len(keys))
    ntrain, nval = int(.6*len(keys)), int(.2*len(keys))
    split = {name: np.array([i for k in ids for i in groups[keys[k]]], dtype=np.int64)
             for name, ids in [('train',order[:ntrain]), ('validation',order[ntrain:ntrain+nval]),
                               ('test',order[ntrain+nval:])]}
    np.savez(out/'splits.npz', **split)
    # Training-only scale; VT's original coordinate scale is adapted explicitly.
    scale = float(np.std(distances[split['train']]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('training distances have no variation')
    provenance = dict(data_sha256=sha256(args.data), dino_weights_sha256=sha256(args.dino_weights),
                      dino_model='dinov2_vits14', preprocessing='BHW/BCHW [0,1], gray->RGB, bicubic224, ImageNet')
    if args.feature_cache:
        cached = torch.load(args.feature_cache, map_location='cpu', weights_only=False)
        if cached['provenance'] != provenance:
            raise ValueError('feature cache provenance mismatch')
        features = cached['features']
    else:
        backbone = FrozenDinoV2(args.dino_repo, args.dino_weights, args.device)
        features = []
        for start in range(0, len(images), 16):
            features.append(backbone(torch.from_numpy(images[start:start+16]).to(args.device)).cpu())
            if start % 80 == 0:
                print(f'DINO features: {min(start+16,len(images))}/{len(images)}', flush=True)
        features = torch.cat(features)
    if features.shape != (len(images),384) or not torch.isfinite(features).all():
        raise ValueError('invalid DINO feature cache')
    torch.save(dict(provenance=provenance, features=features), out/'features.pt')
    # Extraction constructs a backbone and consumes RNG; cache reuse must not
    # change projection initialization or its minibatch order.
    torch.manual_seed(args.seed)
    model = ObservationProjection().to(args.device)
    train = torch.as_tensor(split['train'], device=args.device)
    val = torch.as_tensor(split['validation'], device=args.device)
    features = features.to(args.device)
    model.feature_mean.copy_(features[train].mean(0,keepdim=True))
    model.feature_std.copy_(features[train].std(0,keepdim=True).clamp_min(.01))
    target = torch.tensor(distances/scale, device=args.device)[:,None]
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, best_epoch = float('inf'), None, None
    for epoch in range(args.epochs):
        model.train()
        for batch in train[torch.randperm(len(train), device=args.device)].split(64):
            # Scalar Euclidean observation matching, as in upstream loss_uni.
            loss = (model(features[batch])-target[batch]).norm(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float((model(features[val])-target[val]).square().mean())
        if val_loss < best:
            best, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch+1
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        observations = model(features).cpu().numpy().reshape(-1)
    checkpoint = dict(format='vision_sls_aebs_v1', projection=model.cpu().state_dict(),
                      feature_dim=384, hidden_dim=128, distance_scale_m=scale,
                      dino_repo=str(args.dino_repo.resolve()), dino_weights=str(args.dino_weights.resolve()),
                      provenance=provenance, best_epoch=best_epoch, seed=args.seed)
    torch.save(checkpoint, out/'observation.pt')
    np.savez(out/'observations.npz', distance_m=distances, observation=observations)
    metrics = {}
    for name, ids in split.items():
        error = observations[ids]-distances[ids]/scale
        metrics[name] = dict(samples=len(ids), rmse_m=float(np.sqrt(np.mean(error**2))*scale),
                             mae_m=float(np.mean(np.abs(error))*scale),
                             residual_linf_max=float(np.abs(error).max()))
    return checkpoint, images, distances, observations, split, metrics


def evaluate(policy, scale, distances=None, observations=None):
    return rollout(policy,scale,distances=distances,observations=observations)


def run_vt(args, policy, out, task_panel):
    from Aebs.system.env import Aebs
    from Aebs.vision_sls.training import TaskPreservingLearner
    from Aebs.VT.verify import VTVerifier
    from Aebs.VT.loop import Loop
    env = Aebs(.05)
    if args.smoke:
        env.space_split = env.train_space_split = np.array([6,6])
    policy.set_state_distance_scale(env.std1)
    learner = TaskPreservingLearner([2,16,8,1], env, p_lip=2., l_lip=4., eps=.1,
                        gamma_decrease=1., reach_prob=.95, square_l_output=True, p_net=policy,
                        task_panel=task_panel,actor_learning_rate=args.actor_learning_rate)
    before_barrier = copy.deepcopy(learner.l_model.state_dict())
    before_actor = copy.deepcopy(policy.controller_net.state_dict())
    bounded = learner.create_bounded_module(learner.l_model)
    verifier = VTVerifier(learner,env,bounded,batch_size=2048,reach_prob=.9,fail_check_fast=True)
    loop = Loop(learner,verifier,env,jitter_grid=env.space_split,soft_constraint=False,
                max_iteration_index=args.max_iteration_index)
    loop.run(args.vt_seconds)
    (out/'actor_update_history.json').write_text(json.dumps(learner.task_update_history,indent=2))
    torch.save(learner.l_model.state_dict(),out/'sbc.pt')
    # Check the SAVED actor/barrier pair after the final actor update.
    final_verifier = VTVerifier(learner,env,learner.create_bounded_module(learner.l_model),
                                batch_size=2048,reach_prob=.9,fail_check_fast=False)
    final_verifier.prefill_train_buffer()
    sat, hard, info, _ = final_verifier.check_dec_cond(1.2)
    _, init_upper = final_verifier.compute_bound_init(env.space_split)
    unsafe_lower, _ = final_verifier.compute_bound_unsafe(env.space_split)
    domain_lower, _ = final_verifier.compute_bound_domain(env.space_split)
    return dict(loop_info=loop.info, final_nominal_decrease_satisfied=bool(sat),
                final_nominal_hard_violations=int(hard), final_info=info,
                final_bounds=dict(init_upper=float(init_upper), unsafe_lower=float(unsafe_lower),
                                  domain_lower=float(domain_lower)),
                barrier_updated=any(not torch.equal(v,learner.l_model.state_dict()[k]) for k,v in before_barrier.items()),
                actor_updated=any(not torch.equal(v,policy.controller_net.state_dict()[k]) for k,v in before_actor.items()),
                actor_learning_rate=args.actor_learning_rate,
                actor_updates_accepted=sum(r['accepted'] for r in learner.task_update_history),
                actor_updates_rejected=sum(not r['accepted'] for r in learner.task_update_history),
                image_closed_loop_certified=False, grid=env.space_split.tolist())


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    args.ppo_steps = args.ppo_steps if args.ppo_steps is not None else (512 if args.smoke else 30000)
    args.max_iteration_index = args.max_iteration_index if args.max_iteration_index is not None else (0 if args.smoke else 100)
    if min(args.epochs,args.ppo_steps,args.threads,args.vt_seconds,args.actor_learning_rate) <= 0 or args.max_iteration_index < 0 or not np.isfinite(args.actor_learning_rate):
        parser.error('positive budgets and nonnegative iteration index required')
    for path in [args.data,args.dino_weights,args.dino_repo/'hubconf.py']:
        if not path.is_file():
            parser.error(f'missing input: {path}')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output directory must be empty; select --output-dir')
    out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2,default=str))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('Training AEBS VISION-SLS observation: frozen DINO -> 1-D Cx',flush=True)
    ckpt, images, distances, observations, split, perception = train_observation(args,out)
    (out/'perception_metrics.json').write_text(json.dumps(perception,indent=2))
    ids = split['train']
    env = ObservationEnv(ckpt['distance_scale_m'],distances[ids],observations[ids],mixed=True)
    ppo = PPO('MlpPolicy',env,n_steps=256,batch_size=64,seed=args.seed,device='cpu',verbose=0)
    print(f'Training PPO: {args.ppo_steps} steps, nominal/train-image mixed episodes',flush=True)
    ppo.learn(total_timesteps=args.ppo_steps)
    ppo.save(out/'ppo_initial.zip')
    policy = VisionSLSPolicy(CombinedPolicyNetwork(ppo.policy.mlp_extractor.policy_net,ppo.policy.action_net),
                            ckpt['distance_scale_m']).eval()
    from Aebs.vision_sls.training import TrainingTaskPanel
    task_panel = TrainingTaskPanel(ckpt['distance_scale_m'],distances[ids],observations[ids])
    before_vt = task_panel(policy)
    (out/'pre_vt_training_rollouts.json').write_text(json.dumps(before_vt,indent=2))
    # Real DINO image -> projection -> actor, independent of the cached replay.
    encoder = ImageObservation(out/'observation.pt',device=args.device)
    # Interface checks use training images; test images never select updates.
    sample_ids = split['train'][:4]
    with torch.no_grad():
        real_obs = encoder(torch.tensor(images[sample_ids],device=args.device)).cpu()
        expected = torch.tensor(observations[sample_ids])[:,None]
        gap = float((real_obs-expected).abs().max())
        if not torch.allclose(real_obs,expected,atol=2e-4,rtol=2e-4):
            raise RuntimeError('deployment image path disagrees with feature cache')
        actions = policy.forward_observation(real_obs,torch.full_like(real_obs,2.75))
        if not torch.isfinite(actions).all():
            raise RuntimeError('nonfinite image actions')
    print('Running original SPVC learner/verifier with nominal Cx observations',flush=True)
    vt = run_vt(args,policy,out,task_panel)
    policy.cpu().eval()
    torch.save(dict(format='vision_sls_policy_v1',state_dict=policy.state_dict(),
                    observation_sha256=sha256(out/'observation.pt'),metadata=policy.metadata(),
                    actor_hidden=[64,64],activation='Tanh'),out/'policy.pt')
    test = split['test']
    result = dict(frontend='vision_sls',smoke=args.smoke,seed=args.seed,
                  perception=perception,vision_sls_observation=dict(C=[[1.,0.]],dimension=1),
                  image_forward_cache_max_gap=gap,image_forward_actions=actions.tolist(),
                  nominal_rollout=evaluate(policy,ckpt['distance_scale_m']),
                  heldout_image_replay=evaluate(policy,ckpt['distance_scale_m'],distances[test],observations[test]),
                  spvc=vt,policy=policy.metadata(),pre_vt_training_rollouts=before_vt,
                  claim='Pipeline execution only. Nominal VT does not certify image residuals; replay is not live CARLA.',
                  artifacts={p.name:sha256(p) for p in [out/'observation.pt',out/'policy.pt',out/'sbc.pt']})
    (out/'metrics.json').write_text(json.dumps(result,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    print(json.dumps(result,indent=2,default=str),flush=True)


if __name__ == '__main__':
    main()
