"""Guard against mismatches between the six-state implementation and its certificate."""
from itertools import product
import unittest

import numpy as np

from Aebs.visual_hover import certificate, plant


class HoverCertificateTest(unittest.TestCase):
    design = dict(l=.7, m=1., s_limit=.63)

    def test_transformed_dynamics_and_barrier_at_corners(self):
        q = .35
        self.assertTrue(certificate.verify(self.design, q)['passed'])
        corners = np.array(list(product([-1, 1], repeat=4)), dtype=float)
        l, m, S = self.design.values()
        for r in [1., 1.12, 1.25]:
            p, s, eta, wind = [np.repeat(col[:,None], 3, axis=1) for col in
                               (r*corners[:,0], r*S*corners[:,1], q*corners[:,2], plant.WIND*corners[:,3])]
            v = s-l*p
            state = np.column_stack((p,v))
            action = plant.control(p+eta,v,self.design)
            actual = plant.step(state,action,wind)
            next_p, next_v = np.split(actual,2,axis=1)
            np.testing.assert_allclose(next_v+l*next_p,(1-plant.DT*m)*s+plant.DT*(wind-l*m*eta))
            self.assertTrue(np.all(certificate.barrier(actual,self.design) <=
                                   certificate.barrier(state,self.design) + 1e-12))

    def test_too_large_radius_has_actual_increase(self):
        q = 1.1
        audit = certificate.verify(self.design,q)
        self.assertFalse(audit['passed'])
        self.assertLess(audit['margins']['transformed_velocity_update'],0)
        state = np.array([0.,0,0,self.design['s_limit'],0,0])
        action = plant.control(np.array([-q,0,0]),state[3:],self.design)
        nxt = plant.step(state,action,np.array([plant.WIND,0,0]))
        self.assertEqual(certificate.barrier(state,self.design),0)
        self.assertGreater(certificate.barrier(nxt,self.design),0)

    def test_initial_and_unsafe_thresholds(self):
        initial = np.array(list(product(*([[-plant.INIT_POS,plant.INIT_POS]]*3 +
                                          [[-plant.INIT_VEL,plant.INIT_VEL]]*3))))
        self.assertLessEqual(certificate.barrier(initial,self.design).max(),1)
        for axis,limit in enumerate([plant.POS_LIMIT]*3+[plant.VEL_LIMIT]*3):
            for sign in [-1,1]:
                unsafe = np.zeros(6)
                unsafe[axis] = sign*limit
                self.assertGreaterEqual(certificate.barrier(unsafe,self.design),20)


if __name__ == '__main__':
    unittest.main()
