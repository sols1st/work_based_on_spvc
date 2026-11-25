"""Train the released SPVC sources on the shared, reformatted RGB training split."""
import argparse
import hashlib
import os
import random
import runpy
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .aebs_matched_data import write_json

MODULES = {
    'ppo': 'Aebs.controller.Controller_train',
    'generator': 'Aebs.cGAN.train_mlp',
    'state': 'Aebs.controller.StateEstimate_train',
}
RELEASE = '6279872ed7708cd103c0801d758d2a95c203ce27'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cap_ppo_evaluation(steps):
    """Bound evaluation callback episodes while leaving PPO rollouts unchanged."""
    from gymnasium.wrappers import TimeLimit
    from stable_baselines3.common import callbacks

    original = callbacks.EvalCallback

    class BoundedEvalCallback(original):
        def __init__(self, eval_env, *args, **kwargs):
            super().__init__(TimeLimit(eval_env, max_episode_steps=steps), *args, **kwargs)

    callbacks.EvalCallback = BoundedEvalCallback


def synthesize(out, rounds):
    from Aebs.system.env import Aebs
    from Aebs.VT.loop import Loop
    from Aebs.VT.train import VTLearner
    from Aebs.VT.verify import VTVerifier

    env = Aebs(.05)
    learner = VTLearner([2, 16, 8, 1], env, 2., 4., .1, 1., .95, square_l_output=True)
    verifier = VTVerifier(learner, env, learner.create_bounded_module(learner.l_model),
                          batch_size=2048, reach_prob=.9, fail_check_fast=True)
    loop = Loop(learner, verifier, env, jitter_grid=env.space_split, soft_constraint=False)
    rows = []
    original_train = loop.train
    original_check = verifier.check_dec_cond
    verdict = {}

    def check(k):
        result = original_check(k)
        verdict.update(accepted=bool(result[0]), hard_violations=int(result[1]))
        return result

    def save(name):
        torch.save(dict(policy=learner.p_net.state_dict(), barrier=learner.l_model.state_dict(),
                        iteration=loop.iter), out / (name + '.pt'))

    def train(which, *args, **kwargs):
        if which == 'p':
            row = dict(iteration=loop.iter, **verdict)
            if row['accepted']:
                row.update({k: float(loop.info[k]) for k in ('ub_init', 'lb_unsafe', 'domain_min')})
                separation = row['lb_unsafe'] - row['domain_min']
                if separation > 0 and row['lb_unsafe'] >= row['ub_init']:
                    row['score'] = 1 - (row['ub_init'] - row['domain_min']) / separation
                    if row['score'] > max((x.get('score', -np.inf) for x in rows), default=-np.inf):
                        save('best')
            save('last_verified')
            rows.append(row)
            write_json(out / 'rounds.json', rows)
            if len(rows) >= rounds:
                return
        return original_train(which, *args, **kwargs)

    loop.train = train
    verifier.check_dec_cond = check
    # Default budget executes rounds 0 through 100. The exception path stops
    # a deliberately shorter pilot before the next original training round.
    def train_with_limit(which, *args, **kwargs):
        if which == 'l' and len(rows) >= rounds:
            raise EndOfBudget
        return train(which, *args, **kwargs)

    class EndOfBudget(Exception):
        pass

    loop.train = train_with_limit
    try:
        loop.run(3600)
    except EndOfBudget:
        pass
    scored = [row for row in rows if 'score' in row]
    selected = 'best.pt' if scored else 'last_verified.pt'
    write_json(out / 'synthesis.json', dict(
        rounds=len(rows), author_accepted=sum(row['accepted'] for row in rows),
        scored=len(scored), selected=selected, selected_iteration=(max(scored, key=lambda r: r['score'])
                                                                    if scored else rows[-1]),
        selection_rule='best separated native score; final pre-update pair if no candidate scores',
        weights_sha256=sha(out / selected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=[*MODULES, 'synthesis'])
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=251)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=101)
    parser.add_argument('--ppo-eval-steps', type=int)
    args = parser.parse_args()
    root, out = args.upstream.resolve(), args.output.resolve()
    before = {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')}
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / 'auto_LiRPA')]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        original_load = torch.load

        def load(*items, **kw):
            kw.setdefault('map_location', 'cpu')
            return original_load(*items, **kw)

        torch.load = load

    started = time.monotonic()
    if args.stage == 'ppo' and args.ppo_eval_steps:
        cap_ppo_evaluation(args.ppo_eval_steps)
    if args.stage in MODULES:
        runpy.run_module(MODULES[args.stage], run_name='__main__')
        if args.stage == 'generator':
            generated = root / 'Aebs/cGAN/mlp_supervised_test_ld4/mlp_supervised.pth'
            expected = root / 'Aebs/cGAN/mlp_supervised_ld4/mlp_supervised.pth'
            expected.write_bytes(generated.read_bytes())
    else:
        synthesize(out, args.rounds)
    assert before == {str(p.relative_to(root)): sha(p) for p in root.rglob('*.py')}
    write_json(out / (args.stage + '_run.json'), dict(
        release=RELEASE, stage=args.stage, seconds=time.monotonic() - started,
        seed=args.seed, torch=torch.__version__, device='cuda' if torch.cuda.is_available() else 'cpu',
        ppo_evaluation_limit_steps=args.ppo_eval_steps, h5_sha256=sha(root / 'Aebs/data/Downsampled.h5'), source_hashes=before))


if __name__ == '__main__':
    main()
