import numpy as np
import pytest
import torch
from Aebs.vision_sls.conformal import envelope,validate_contract_units
from Aebs.vision_sls.repair_real_experiment import grouped_behavior


def test_normalized_score_requires_metre_scale_and_old_contract_is_rejected():
    s=3.18272066116333;q=.07249307632446289
    broken=dict(coefficients=[1.]*4,calibration=dict(q_hat=q),coverage=dict(calibration=dict(mean_radius_m=q*s)))
    with pytest.raises(ValueError,match='unit mismatch'):validate_contract_units(broken)
    fixed={**broken,'coefficients':[s]*4,'envelope_units':'m'}
    validate_contract_units(fixed)
    np.testing.assert_allclose(envelope([5,10,16],fixed['coefficients'],q),[q*s]*3,rtol=1e-14)
    # physical observation endpoints normalize back to z +/- q, not q/s.
    assert abs((10+q*s)/s-10/s-q)<1e-14


def test_grouped_replay_never_mixes_appearances(monkeypatch):
    import Aebs.vision_sls.repair_real_experiment as module
    seen=[]
    def fake(policy,scale,initial,distance,observation):
        assert len(np.unique(observation))==1
        seen.append(float(observation[0]));return dict(episodes=len(initial),counts={'success':len(initial)},lookup_max_error_m=0.)
    monkeypatch.setattr(module,'rollout',fake)
    library=dict(weather=np.array(['sun','sun','rain','rain']),color=np.array(['red']*4),
        distance_m=np.array([10,10,10,10]),observation=np.array([1,1,2,2]),split=np.array(['train','test']*2))
    out=grouped_behavior(None,3.,library,size=2)
    assert sorted(seen)==[1,1,2,2] and len(out['groups']['all'])==2


@pytest.mark.parametrize('method',['ibp','affine'])
def test_actor_bounds_enclose_physical_observation_tube(method):
    from types import SimpleNamespace
    from Aebs.vision_sls.verify_sbc_cells import actor_box
    torch.manual_seed(23)
    hidden=torch.nn.Sequential(torch.nn.Linear(2,8),torch.nn.Tanh(),torch.nn.Linear(8,4),torch.nn.Tanh()).double()
    head=torch.nn.Linear(4,1).double()
    policy=SimpleNamespace(distance_scale_m=3.18272066116333,
        controller_net=SimpleNamespace(mlp_extractor=hidden,action_net=head))
    lower=torch.tensor([[6.1,.4],[10.,2.]],dtype=torch.float64)
    upper=lower+torch.tensor([.02,.03],dtype=torch.float64)
    radius=torch.tensor([.4183585683531419]*2,dtype=torch.float64)
    al,au=actor_box(policy,lower,upper,radius,8,method)
    for i in range(2):
        obs=torch.linspace(lower[i,0]-radius[i],upper[i,0]+radius[i],101)
        speed=torch.linspace(lower[i,1],upper[i,1],23)
        yy,vv=torch.meshgrid(obs,speed,indexing='ij')
        inputs=torch.stack([yy/policy.distance_scale_m,vv],-1).reshape(-1,2).double()
        values=head(hidden(inputs)).clamp(-3,3)
        assert values.min()>=al[i]-1e-12 and values.max()<=au[i]+1e-12


def test_episode_score_includes_lookup_error_and_retains_failed_episodes(monkeypatch):
    import Aebs.vision_sls.replay_trajectory_cp as module
    def fake(policy,scale,initial,distances,observations,record_trajectories):
        assert record_trajectories
        return dict(trajectories=[dict(indices=[0,1],observation=[4.,4.],distance_m=[11.9,11.8]),
                                  dict(indices=[1],observation=[4.],distance_m=[11.6])],
                    outcomes=['unsafe','timeout'],steps=[1,2],lookup_max_error_m=.4)
    monkeypatch.setattr(module,'rollout',fake)
    # The saved image itself has exact prediction 4*3=12 m; errors arise
    # entirely from querying it at different physical states.
    library=dict(weather=np.array(['sun']),color=np.array(['red']),
                 distance_m=np.array([12.]),observation=np.array([4.]))
    data,lookup=module.collect(None,3.,library,2,41)
    np.testing.assert_allclose(data['scores'],[.1,.4])
    assert data['outcomes'].tolist()==['unsafe','timeout'] and lookup==.4
