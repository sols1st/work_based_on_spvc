import itertools
import numpy as np
import torch
from torch import nn
from Aebs.mobile_robot import core
from Aebs.mobile_robot.bounds import transitions,trig_bounds,unsafe_boxes
from Aebs.mobile_robot.renderer import RobotRenderer
from Aebs.mobile_robot.env import RobotEnv
from Aebs.lane_keeping.policy import DeterministicActor,load_actor,export_actor


def test_dynamics_numpy_torch_agree():
    rng=np.random.default_rng(1);x=rng.uniform(core.LOW,core.HIGH,(100,3));u=rng.uniform(-2,2,(100,2))
    actual=core.physical(core.torch_dynamics(torch.tensor(core.normalize(x)),torch.tensor(u)).numpy())
    np.testing.assert_allclose(actual,core.dynamics(x,u),atol=1e-12)


def test_interval_transition_encloses_two_action_dynamics():
    torch.manual_seed(8);rng=np.random.default_rng(8)
    actor=DeterministicActor(nn.Sequential(nn.Linear(3,8),nn.Tanh(),nn.Linear(8,8),nn.Tanh()),nn.Linear(8,2)).double()
    lo=torch.tensor(rng.uniform(-.9,.8,(20,3)));hi=lo+.1
    el=torch.tensor([-.03,-.04,-.2]);eh=-el
    with torch.no_grad():
        nl,nh=transitions(actor,lo,hi,el,eh)
        for _ in range(50):
            x=lo+(hi-lo)*torch.rand_like(lo);e=el+(eh-el)*torch.rand_like(lo)
            y=core.torch_dynamics(x,actor(x+e))
            assert torch.all(y>=nl-1e-11) and torch.all(y<=nh+1e-11)


def test_periodic_trig_bounds_include_critical_points():
    lo=torch.tensor([-10.,-.2,1.,4.],dtype=torch.float64);hi=lo+3
    for cosine in [False,True]:
        a,b=trig_bounds(lo,hi,cosine)
        points=lo[:,None]+(hi-lo)[:,None]*torch.linspace(0,1,1000)
        values=points.cos() if cosine else points.sin()
        assert torch.all(values>=a[:,None]-1e-12) and torch.all(values<=b[:,None]+1e-12)


def test_unsafe_cover_contains_disk_and_all_first_exit_slabs():
    rng=np.random.default_rng(4);reach=np.array([.06,.06,.1])
    points=rng.uniform(core.LOW-reach,core.HIGH+reach,(5000,3));points=points[core.unsafe(points)]
    lo,hi=unsafe_boxes();lo=lo.numpy();hi=hi.numpy()
    for x in core.normalize(points):assert ((x>=lo-1e-12)&(x<=hi+1e-12)).all(1).any()


def test_goal_is_not_absorbing_or_terminal():
    env=RobotEnv(1);env.reset();env.state=np.array([5.,2.,0.])
    _,_,terminated,truncated,_=env.step(np.array([1.,0.]))
    assert not terminated and not truncated
    assert env.state[0]>5


def test_heading_visible_in_image():
    renderer=RobotRenderer()
    assert not np.array_equal(renderer.render([1,2,0]),renderer.render([1,2,.7]))


def test_ppo_export_matches_deterministic_two_action_prediction(tmp_path):
    from stable_baselines3 import PPO
    model=PPO('MlpPolicy',RobotEnv(1),n_steps=8,batch_size=8,policy_kwargs=dict(net_arch=[8,8]),seed=6,device='cpu')
    file=tmp_path/'actor.pt';export_actor(model,file);actor=load_actor(file)
    x=np.random.default_rng(3).uniform(-1,1,(100,3)).astype(np.float32)
    expected,_=model.predict(x,deterministic=True)
    with torch.no_grad():actual=actor(torch.tensor(x)).numpy()
    assert actual.shape==(100,2)
    np.testing.assert_allclose(actual,expected,atol=1e-7)


def test_affine_training_has_finite_gradients_at_unit_tanh_slope():
    from Aebs.vision_sls.neural_bounds import tanh_affine_bounds
    layers=[nn.Linear(3,4),nn.Tanh(),nn.Linear(4,1)]
    with torch.no_grad():layers[0].weight.fill_(.1);layers[0].bias.zero_()
    lo=torch.zeros(5,3);hi=torch.zeros(5,3)+1e-9
    normal=tanh_affine_bounds(layers,lo,hi)
    stable=tanh_affine_bounds(layers,lo,hi,detach_root=True)
    torch.testing.assert_close(normal[0],stable[0]);torch.testing.assert_close(normal[1],stable[1])
    stable[1].sum().backward()
    for layer in layers:
        for param in layer.parameters():assert torch.isfinite(param.grad).all()


def test_trajectory_cp_catches_single_heading_frame_and_small_n():
    from Aebs.mobile_robot.trajectory_cp import trajectory_scores
    from Aebs.lane_keeping.cp import conformal_radius
    errors=np.zeros((2,120,3));errors[0,73,2]=.4;errors[1,7,0]=-.09
    np.testing.assert_allclose(trajectory_scores(errors,[.03,.03,.2]),[2,3])
    q,rank=conformal_radius(np.arange(10),.05)
    assert np.isinf(q) and rank==11
    q,rank=conformal_radius(np.arange(39),.05)
    assert q==37 and rank==38


def test_anisotropic_error_partition_covers_registered_box():
    from Aebs.mobile_robot.bounds import error_boxes
    pairs=error_boxes([.03,.04,.2],2)
    assert len(pairs)==8
    volume=sum(float((hi-lo).prod()) for lo,hi in pairs)
    assert abs(volume-.06*.08*.4)<1e-14
    for corner in itertools.product([-.03,.03],[-.04,.04],[-.2,.2]):
        x=torch.tensor(corner,dtype=torch.float64)
        assert any(bool(((x>=lo)&(x<=hi)).all()) for lo,hi in pairs)
