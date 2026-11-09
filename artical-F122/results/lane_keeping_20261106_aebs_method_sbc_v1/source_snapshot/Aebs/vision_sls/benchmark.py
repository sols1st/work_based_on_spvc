"""Evaluate frozen checkpoints on a dense grid and fresh random initial states.

No policy update or model selection occurs here. Uses the original initial
rectangle [15,16] m x [2.5,3] m/s and the unchanged AEBS success event.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from Combined_network.model import CombinedPolicyNetwork
from .deploy import VisionSLSController
from .model import VisionSLSPolicy
from .rollout import initial_grid, random_initial, rollout
from .run import ROOT, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--grid-size',type=int,default=21)
    parser.add_argument('--random-episodes',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=20261028)
    args = parser.parse_args(argv)
    if args.grid_size < 2 or args.random_episodes < 1:
        parser.error('grid size >=2 and positive random episode count required')
    torch.set_num_threads(4)
    root = args.artifact_dir
    controller = VisionSLSController(root,'cpu')
    data = np.load(root/'observations.npz')
    ids = np.load(root/'splits.npz')['test']
    checkpoint = torch.load(root/'observation.pt',map_location='cpu',weights_only=False)
    ppo = PPO.load(root/'ppo_initial.zip',device='cpu')
    initial_policy = VisionSLSPolicy(CombinedPolicyNetwork(ppo.policy.mlp_extractor.policy_net,ppo.policy.action_net),
                                    checkpoint['distance_scale_m'])
    results = {}
    for stage,policy in [('before_vt',initial_policy),('after_vt',controller.policy)]:
        results[stage] = {}
        for panel,states in [('grid',initial_grid(args.grid_size)),
                             ('random',random_initial(args.random_episodes,args.seed))]:
            nominal = rollout(policy,checkpoint['distance_scale_m'],states)
            replay = rollout(policy,checkpoint['distance_scale_m'],states,
                              data['distance_m'][ids],data['observation'][ids])
            results[stage][panel] = dict(nominal=nominal,heldout_images=replay)
            print(stage,panel,nominal['counts'],replay['counts'],flush=True)
    # Short inspectable witness rather than millions of trajectory coordinates.
    witness = np.array([[15.5,2.75]])
    trace = {stage:rollout(policy,checkpoint['distance_scale_m'],witness,
                          data['distance_m'][ids],data['observation'][ids],record_trajectories=True)
             for stage,policy in [('before_vt',initial_policy),('after_vt',controller.policy)]}
    result = dict(seed=args.seed,initial_domain=[[15.,16.],[2.5,3.]],
                  policy_sha256=sha256(root/'policy.pt'),observation_sha256=sha256(root/'observation.pt'),
                  training_or_selection=False,live_carla=False,image_closed_loop_certified=False,
                  results=results)
    (root/'expanded_benchmark.json').write_text(json.dumps(result,indent=2))
    (root/'witness_trajectory.json').write_text(json.dumps(trace,indent=2))


if __name__ == '__main__':
    main()
