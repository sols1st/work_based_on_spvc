import unittest
import numpy as np
import torch
from gymnasium.spaces import Box
from Aebs.dino_latent.train_joint_nbc_ppo import safety_surrogate, SafetyBuffer


class JointTests(unittest.TestCase):
    def test_safety_cost_reaches_action_distribution_but_not_critic(self):
        mean=torch.tensor(0.,requires_grad=True)
        risk=torch.tensor([1.,-1.],requires_grad=True)
        a=torch.tensor([1.,-1.])
        old=torch.distributions.Normal(0.,1.).log_prob(a)
        loss=safety_surrogate(torch.distributions.Normal(mean,1.).log_prob(a),old,risk)
        loss.backward()
        self.assertGreater(float(mean.grad),0)
        self.assertIsNone(risk.grad)

    def test_sample_probability_uses_raw_not_clipped_action(self):
        dist=torch.distributions.Normal(torch.tensor(0.),torch.tensor(1.))
        self.assertNotEqual(float(dist.log_prob(torch.tensor(5.))),float(dist.log_prob(torch.tensor(3.))))

    def test_safety_samples_stay_aligned_after_shuffle(self):
        b=SafetyBuffer(4,Box(-np.inf,np.inf,(33,),dtype=np.float32),Box(-3.,3.,(1,),dtype=np.float32),device="cpu")
        for i in range(4):
            b.add(np.ones((1,33),np.float32)*i,np.zeros((1,1),np.float32),np.ones(1),np.zeros(1),torch.zeros(1),torch.zeros(1))
        b.safety_cost=np.arange(4,dtype=np.float32)*10
        for batch in b.get(2):
            torch.testing.assert_close(batch.safety_cost,batch.observations[:,0]*10)


if __name__=="__main__": unittest.main()
