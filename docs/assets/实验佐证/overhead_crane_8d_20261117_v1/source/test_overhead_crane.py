"""Check coupling, exact bounds, and the time-dependent crane barrier independently."""
from itertools import product
import unittest
import numpy as np

from Aebs.overhead_crane import certificate,plant


class CraneCertificateTest(unittest.TestCase):
    gains=[1.7103,2.3227,-2.4583,-.4332]

    def test_dynamics_and_motor_budget(self):
        rng=np.random.default_rng(8)
        state=rng.uniform(-.3,.3,(100,2,4))
        noise=rng.uniform(-1,1,(100,2,2))*[plant.CART_NOISE,plant.LOAD_NOISE]
        action=rng.uniform(-3,3,(100,2))
        actual=plant.step(state,action,noise)
        p,v,s,r=np.moveaxis(state,-1,0)
        v1=v+plant.DT*(action+noise[...,0])
        r1=r+plant.DT*(-4*s-.16*r-action-noise[...,0]+noise[...,1])
        expected=np.stack((p+plant.DT*v1,v1,s+plant.DT*r1,r1),axis=-1)
        np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
        z=np.full((1,4),100.)
        limited=plant.control(z,np.zeros((1,2,4)),self.gains)
        np.testing.assert_allclose(abs(limited).sum(1),[plant.TOTAL_ACCEL])
        self.assertLessEqual(abs(limited).max(),plant.AXIS_ACCEL)

    def test_exact_initial_bound_and_rejection(self):
        q=.12
        audit=certificate.verify(self.gains,q)
        self.assertTrue(audit['passed'])
        corners=np.array(list(product([-1,1],repeat=8))).reshape(-1,2,4)*plant.INITIAL
        self.assertAlmostEqual(certificate.barrier(corners,self.gains,q,plant.HORIZON).max(),
                               audit['initial_upper'],places=12)
        self.assertFalse(certificate.verify(self.gains,1.)['passed'])
        # Both payloads within each axis bound but the diagonal clearance is hit.
        corner=np.array([[[1.10,0.,.20,0.],[1.10,0.,.20,0.]]])
        self.assertGreaterEqual(certificate.barrier(corner,self.gains,q,0)[0],1)

    def test_all_corner_successors_and_one_step_definition(self):
        q=.12
        state=np.array([[[.33,.07,.05,-.08],[-.51,-.02,-.09,.11]]])
        signs=np.array(list(product([-1,1],repeat=8)))
        z=plant.observation(state)+q*signs[:,:4]
        noise=signs[:,4:].reshape(-1,2,2)*[plant.CART_NOISE,plant.LOAD_NOISE]
        action=plant.control(z,state,self.gains)
        nxt=plant.step(state,action,noise)
        now=certificate.barrier(state,self.gains,q,1)[0]
        # Independently maximize all present motor/state constraints and next hazards.
        motor=max(abs(action).max()/plant.AXIS_ACCEL,abs(action).sum(1).max()/plant.TOTAL_ACCEL)
        direct=max(motor,certificate.barrier(state,self.gains,q,0).max(),
                   certificate.barrier(nxt,self.gains,q,0).max())
        self.assertAlmostEqual(now,direct,places=12)
        for remaining in [2,10,plant.HORIZON]:
            before=certificate.barrier(state,self.gains,q,remaining)[0]
            self.assertLess(before,1)
            self.assertLessEqual(certificate.barrier(nxt,self.gains,q,remaining-1).max(),before+1e-12)


if __name__=='__main__':
    unittest.main()
