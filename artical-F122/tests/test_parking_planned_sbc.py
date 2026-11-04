import unittest
from types import SimpleNamespace
import numpy as np
import torch
from Aebs.vision_sls_parking.core import STATE_LOW, STATE_HIGH, OBSTACLES, constraints
from Aebs.vision_sls_parking.planned_sbc import risk, PlannedBarrier, finite_horizon_barrier


class PlannedSBCTests(unittest.TestCase):
    def test_geometric_unsafe_lower_bound_includes_collision_and_bound_tolerances(self):
        samples=[]
        for axis in range(4):
            for edge,sign in [(STATE_LOW[axis],-1),(STATE_HIGH[axis],1)]:
                x=np.array([0.,0.,0.,0.]);x[axis]=edge+sign*.00101;samples.append(x)
        for px,py,r in OBSTACLES:
            x=np.array([px+np.sqrt(r*r+.0001)-1e-8,py,0.,0.]);samples.append(x)
        self.assertTrue(all(constraints(x)['collision'] or not constraints(x)['in_bounds'] for x in samples))
        value=risk(torch.tensor(np.asarray(samples)))
        self.assertTrue((value>=20).all())

    def test_nonnegative_head_keeps_unsafe_bound_and_rejects_invalid_clock(self):
        bank=SimpleNamespace(states=np.zeros((15,31,4)))
        barrier=PlannedBarrier(bank).double()
        x=torch.randn(400,8,dtype=torch.float64)*4
        j=torch.arange(400)%15;t=torch.arange(400)%31
        v=barrier(x,j,t)
        self.assertTrue((v>=risk(x)).all())
        self.assertTrue((v>=0).all())
        with self.assertRaises(ValueError):barrier(x,j,t+31)

    def test_lookahead_is_suffix_max_and_terminal_equals_risk(self):
        # A deliberately unsafe deterministic flow tests anticipation, not
        # just zero-valued certificates on already safe reference trajectories.
        def flow(state, reference, time):
            step=state.new_tensor([.04,0.,0.,.02,0.,0.,0.,0.])
            return state+step
        state=torch.tensor([[.2,0.,0.,0.,0.,0.,0.,0.]],dtype=torch.float64)
        for time in [0,20,29]:
            b=finite_horizon_barrier(flow,state,0,time)
            bn=finite_horizon_barrier(flow,flow(state,0,time),0,time+1)
            torch.testing.assert_close(b,torch.maximum(risk(state),bn),atol=1e-10,rtol=1e-10)
            self.assertTrue((bn<=b+1e-10).all())
        torch.testing.assert_close(finite_horizon_barrier(flow,state,0,30),risk(state))
        self.assertGreater(float(finite_horizon_barrier(flow,state,0,0)),20)
        with self.assertRaises(ValueError):finite_horizon_barrier(flow,state,0,-1)


if __name__=='__main__':unittest.main()
