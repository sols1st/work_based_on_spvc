import unittest
from itertools import product

import numpy as np

from Aebs.experiments import compare_crane_contracts as grid
from Aebs.overhead_crane import certificate, plant


class CraneContractComparisonTest(unittest.TestCase):
    gains = [1.7103, 2.3227, -2.4583, -.4332]

    def test_zero_steering_error_still_bounds_observations(self):
        errors = np.array([[[.2, .0, -.4*self.gains[0]/self.gains[2], 0.]]])
        cube = {'weights':[1.]*4, 'joint':False}
        joint = dict(cube, joint=True)
        np.testing.assert_allclose(grid.trajectory_scores(errors, self.gains, joint),
                                   .5*grid.trajectory_scores(errors, self.gains, cube))

    def test_uniform_contract_matches_original_certificate(self):
        q = .173
        cube = {'weights':[1.]*4, 'joint':False}
        v = grid.certify(self.gains, q, cube, 1.)
        orig = certificate.verify(self.gains, q)
        self.assertEqual(v['upper_exact'], orig['initial_upper_exact'])

    def test_two_axis_command_with_shared_actuator(self):
        q = .12
        scale = 1.5
        definition = {'weights':[.35,.27,1.,.86], 'joint':False}
        states = np.array(list(product((-1,1),repeat=8))).reshape(-1,2,4)*np.array(plant.INITIAL)*scale
        err = np.array(list(product((-1,1),repeat=4)))*definition['weights']*q
        nominal = states@self.gains
        disturbance = self.gains[0]*err[:,:2] + .5*self.gains[2]*err[:,2:]
        summed = np.abs(nominal[:,None,:]+disturbance).sum(-1).max()
        bound = grid.certify(self.gains, q, definition, scale)['constraints']['total_accel']
        self.assertEqual(bound['step'],0)
        self.assertAlmostEqual(bound['ratio']*plant.TOTAL_ACCEL, summed, places=12)


if __name__ == '__main__':
    unittest.main()
