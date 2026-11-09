"""Regression tests for actor identity and interval containment, not sampled proofs."""
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from Aebs.lane_keeping import core
from Aebs.lane_keeping.policy import load_actor
from Aebs.lane_keeping.certify_linear import lqr, evaluate_cells, quad_bounds, initial_bounds
from Aebs.lane_keeping.certify_visual import evaluate
from Aebs.lane_keeping.certify_neural import (
    actor_affine_bounds, transition_bounds, PolyhedralSBC, geometry)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'results/lane_keeping_20261105'
torch.set_num_threads(2)


@pytest.mark.parametrize('name', ['ppo_semantic', 'ppo_image'])
def test_export_is_identical_to_sb3(name):
    original = PPO.load(PACKAGE/name/'ppo_model.zip', device='cpu')
    actor = load_actor(PACKAGE/name/'actor.pt')
    points = np.random.default_rng(7).uniform(-1.2, 1.2, (1000, 3)).astype('float32')
    with torch.no_grad():
        actual = actor(torch.from_numpy(points)).numpy()
    np.testing.assert_allclose(actual, original.predict(points, deterministic=True)[0],
                               atol=3e-7, rtol=1e-6)


def test_reversed_intervals_are_rejected():
    with pytest.raises(ValueError):
        quad_bounds(torch.eye(2), torch.ones(1, 2), torch.zeros(1, 2))


def test_legacy_negative_feedback_bound_contains_corner_transitions():
    K, P, _ = lqr()
    kt, pt = torch.tensor(K), torch.tensor(P)
    lo = torch.tensor([[-.10, -.15], [.02, .03]], dtype=torch.float64)
    hi = lo + .03
    upper = evaluate_cells(lo, hi, .01, kt, pt)[2]
    flat_upper = evaluate(lo, hi, .01, kt, pt, (.08, .08))[2]
    for a in [0, 1]:
        for b in [0, 1]:
            x = torch.stack([lo[:, 0] if a == 0 else hi[:, 0],
                             lo[:, 1] if b == 0 else hi[:, 1]], 1)
            for de in [-.08, .08]:
                for dt in [-.08, .08]:
                    observed = x + torch.tensor([de, dt])
                    u = (-(observed @ kt) + np.arctan(2.6*.01)/.45).clamp(-1, 1)
                    nxt = core.dynamics_batch(x.numpy()*[.85, .3], u.numpy()*.45,
                                               np.full(len(x), .01))/[.85, .3]
                    q = np.einsum('ni,ij,nj->n', nxt, P, nxt)
                    assert np.all(np.maximum(q-.5, 0) <= flat_upper.numpy()+1e-12)
            u = (-(x @ kt) + np.arctan(2.6*.01)/.45).clamp(-1, 1)
            nxt = core.dynamics_batch(x.numpy()*[.85, .3], u.numpy()*.45,
                                      np.full(len(x), .01))/[.85, .3]
            q = np.einsum('ni,ij,nj->n', nxt, P, nxt)
            assert np.all(q <= upper.numpy()+1e-12)
    assert initial_bounds(pt) > .99  # legacy visual report had hard-coded .99


def test_affine_network_and_dynamics_enclose_actual_samples():
    rng = np.random.default_rng(104)
    actor = load_actor(PACKAGE/'ppo_image/actor.pt').double()
    lo = torch.tensor(rng.uniform(-.8, .6, (90, 3)))
    hi = lo + torch.tensor(rng.uniform(.001, .3, (90, 3)))
    with torch.no_grad():
        al, ah = actor_affine_bounds(actor, lo, hi)
        for _ in range(20):
            point = lo + torch.tensor(rng.uniform(size=lo.shape))*(hi-lo)
            out = actor(point).flatten()
            assert torch.all(out >= al-1e-12) and torch.all(out <= ah+1e-12)
        low, high = lo[:, :2], hi[:, :2]
        nl, nh = transition_bounds(actor, low, high, -.01,
                                    torch.tensor([-.08, -.08]), torch.tensor([.08, .08]))
        for _ in range(20):
            state = low + torch.tensor(rng.uniform(size=low.shape))*(high-low)
            error = torch.tensor(rng.uniform(-.08, .08, low.shape))
            obs = torch.cat([state+error, -torch.ones(len(state), 1)], 1)
            action = actor(obs).flatten().numpy()*.45
            nxt = torch.tensor(core.dynamics_batch(state.numpy()*[.85, .3], action,
                                                    np.full(len(state), -.01))/[.85, .3])
            assert torch.all(nxt >= nl-1e-10) and torch.all(nxt <= nh+1e-10)


def test_relu_network_equals_max_facet_certificate():
    _, P, _ = lqr()
    angles = np.arange(32)*2*np.pi/32
    facets = np.c_[np.cos(angles), np.sin(angles)]@np.linalg.cholesky(P).T
    barrier = PolyhedralSBC(facets)
    x = torch.tensor(np.random.default_rng(8).uniform(-2, 2, (1000, 2)))
    reference = 200*torch.relu((x@barrier.facets.T).amax(1)-1)
    torch.testing.assert_close(barrier(x).flatten(), reference)
    geo = geometry(facets, 1., 200.)
    assert geo['initial_B_upper'] == 0
    assert geo['unsafe_B_lower'] > 20
    assert max(geo['active_sublevel_coordinate_bounds']) < 1
