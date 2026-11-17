import copy
import torch
from torch import nn
from Aebs.lane_keeping.policy import DeterministicActor
from Aebs.lane_keeping.learned_sbc import NonnegativeSBC
from Aebs.lane_keeping.refine_actor import feedback_loss


def test_robust_tracking_can_update_actor_on_flat_barrier():
    torch.manual_seed(16)
    actor=DeterministicActor(nn.Sequential(nn.Linear(3,8),nn.Tanh()),nn.Linear(8,1))
    original=copy.deepcopy(actor).requires_grad_(False)
    b=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu').requires_grad_(False)
    with torch.no_grad():
        for p in b.parameters(): p.zero_()
    x=torch.tensor([[.2,.1],[-.3,.1],[.1,-.2]],dtype=torch.float32)
    loss,safety=feedback_loss(actor,original,b,x,torch.zeros(3),.10,task_weight=.01)
    assert abs(float(safety))<1e-10 and loss>0  # numerical bound padding
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters())
    assert sum(float(p.grad.abs().sum()) for p in actor.parameters())>0


def test_detached_extremum_preserves_bound_values():
    from Aebs.lane_keeping.learned_sbc import bounds, transitions
    torch.manual_seed(17)
    actor=DeterministicActor(nn.Sequential(nn.Linear(3,8),nn.Tanh()),nn.Linear(8,1)).double()
    b=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu').double()
    x=torch.randn(10,2,dtype=torch.float64)*.3
    el=torch.tensor([-.1,-.1]); eh=-el
    plain=transitions(actor,x,x,torch.zeros(10),el,eh)
    detached=transitions(actor,x,x,torch.zeros(10),el,eh,detach_root=True)
    for a,c in zip(plain,detached): assert torch.equal(a,c)
    for a,c in zip(bounds(b,*plain),bounds(b,*plain,detach_root=True)): assert torch.equal(a,c)


def test_raw_region_training_recovers_dead_relu():
    from Aebs.vision_sls.train_sbc_cp import raw_barrier
    b=NonnegativeSBC([2,16,8,1],nonnegative_activation='relu')
    with torch.no_grad():
        b.layers[-1].weight.zero_();b.layers[-1].bias.fill_(-1.)
    x=torch.ones(4,2)
    assert torch.equal(b(x),torch.zeros(4,1))
    loss=torch.relu(25-raw_barrier(b,x)).square().mean()
    loss.backward()
    assert torch.isfinite(b.layers[-1].bias.grad).all()
    assert b.layers[-1].bias.grad.item()<0
