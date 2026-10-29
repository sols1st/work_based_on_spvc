"""Small SPVC actor updates with train-only task-retention checks.

This is an empirical acceptance check, not a safety certificate. The original
barrier/actor losses and dynamics remain in VTLearner. Rejected proposals
restore both actor weights and Adam moments, while retaining barrier training.
"""
import copy
import torch
from Aebs.VT.train import VTLearner
from .rollout import rollout, random_initial


def retention_failures(before, proposed):
    failures = []
    for mode in before:
        previous = before[mode]['outcomes']
        candidate = proposed[mode]['outcomes']
        if len(previous) != len(candidate):
            raise ValueError('retention panels must match')
        if any(a == 'success' and b != 'success' for a,b in zip(previous,candidate)):
            failures.append(mode+':lost_success')
        if any(a != 'unsafe' and b == 'unsafe' for a,b in zip(previous,candidate)):
            failures.append(mode+':new_unsafe_episode')
    return failures


class TrainingTaskPanel:
    def __init__(self, scale, distances, observations, seed=1729, episodes=32):
        self.scale = scale
        self.distances = distances
        self.observations = observations
        # Fixed training initial states; independent of held-out test rollouts.
        self.initial = random_initial(episodes,seed)

    def __call__(self, policy):
        return dict(nominal=rollout(policy,self.scale,self.initial),
                    train_images=rollout(policy,self.scale,self.initial,self.distances,self.observations))


class TaskPreservingLearner(VTLearner):
    def __init__(self, *args, task_panel, actor_learning_rate=3e-4, **kwargs):
        super().__init__(*args,**kwargs)
        if not 0 < actor_learning_rate < float('inf'):
            raise ValueError('positive finite actor learning rate required')
        self.p_optimizer = torch.optim.Adam(self.p_net.controller_net.parameters(),lr=actor_learning_rate)
        self.task_panel = task_panel
        self.task_update_history = []

    def train_epoch(self, *args, **kwargs):
        # Loop always supplies train_fn by keyword. Other modes are unchanged.
        if kwargs.get('train_fn') != 'p':
            return super().train_epoch(*args,**kwargs)
        before_weights = copy.deepcopy(self.p_net.state_dict())
        before_optimizer = copy.deepcopy(self.p_optimizer.state_dict())
        before = self.task_panel(self.p_net)
        metrics = super().train_epoch(*args,**kwargs)
        finite = all(torch.isfinite(p).all() for p in self.p_net.parameters())
        try:
            proposed = self.task_panel(self.p_net) if finite else None
        except ValueError:
            proposed = None
        failures = ['nonfinite_proposal'] if proposed is None else retention_failures(before,proposed)
        if failures:
            self.p_net.load_state_dict(before_weights)
            self.p_optimizer.load_state_dict(before_optimizer)
        record = dict(accepted=not failures,reasons=failures,before=before,proposed=proposed,
                      actor_and_optimizer_restored=bool(failures),test_used=False,
                      check='finite train-only task retention, not a certificate')
        self.task_update_history.append(record)
        print('Actor task-retention:', 'accepted' if not failures else 'rejected: '+','.join(failures),flush=True)
        return metrics
