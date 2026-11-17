"""Registration and numerical interval checks for the action-aware lane contract."""
import unittest
import numpy as np
import torch
from Aebs.lane_contract_comparison.collect import parameters,scores,maxima
from Aebs.lane_contract_comparison.audit import allowed_action,actor_interval
from Aebs.lane_keeping.policy import DeterministicActor
from Aebs.lane_keeping import core


class LaneContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17032)
        self.actor=DeterministicActor(torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Tanh()),
                                     torch.nn.Linear(4,1)).double()
        with torch.no_grad():
            self.actor.action_net.weight.mul_(8)

    def test_interval_intersection_and_extra_order(self):
        lows=torch.tensor([[.6],[-.8],[.4]],dtype=torch.float64)
        highs=torch.tensor([[1.],[-.6],[.8]],dtype=torch.float64)
        nominal=(torch.tensor([[.8],[-.1],[.5]]),torch.tensor([[1.],[.1],[.6]]))
        lo,hi,valid=allowed_action(lows,highs,nominal,.1,.05)
        self.assertTrue(torch.equal(valid,torch.tensor([True,False,True])))
        np.testing.assert_allclose(lo[valid].flatten(),[.65,.35],atol=2e-7)
        np.testing.assert_allclose(hi[valid].flatten(),[1,.75],atol=2e-7)

    def test_joint_score_matches_pathwise_definition(self):
        scale={'visual':[.2,.07],'action':.3}
        rng=np.random.default_rng(2026)
        z=rng.normal(0,.3,(6,5,2));eta=rng.uniform(-.15,.15,(6,5,2))
        curv=np.tile((np.arange(6)%3-1)*.01,(5,1)).T
        def evaluate(x):
            data=np.concatenate([x,curv[:,:,None]/core.KAPPA_SCALE],-1).reshape(-1,3)
            return self.actor(torch.tensor(data)).detach().reshape(6,5).numpy()
        act=evaluate(z+eta)
        data=dict(states=np.concatenate([z,z[:,-1:]],1)*[core.E_SCALE,core.THETA_SCALE],
                  observations=z+eta,errors=eta,actions=act,kappas=curv[:,0])
        s=scores(maxima(data,self.actor),scale)
        for name,value in s.items():
            q=float(np.median(value))
            arm=parameters(name,q,scale)
            inside=(abs(eta)<=arm['width']).all((1,2))
            if arm['action'] is not None:
                inside &= (np.abs(act-evaluate(z))<=arm['action']).all(1)
            np.testing.assert_array_equal(inside,value<=q)

    def test_interval_covers_actual_clipped_actions(self):
        low=torch.tensor([[-.3,-.15]],dtype=torch.float64)
        high=torch.tensor([[.2,.35]],dtype=torch.float64)
        lo,hi=actor_interval(self.actor,low,high,core.KAPPAS[0])
        obs=low+(high-low)*torch.rand(256,2,dtype=low.dtype)
        curve=torch.full((len(obs),1),core.KAPPAS[0]/core.KAPPA_SCALE,dtype=low.dtype)
        val=self.actor(torch.cat([obs,curve],1))
        self.assertTrue(((lo<=val)&(val<=hi)).all())


if __name__=='__main__':
    unittest.main()
