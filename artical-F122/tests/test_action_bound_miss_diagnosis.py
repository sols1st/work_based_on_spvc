import unittest
import numpy as np

from Aebs.dino_latent.diagnose_action_bound_misses import bracket_bounds, miss_partition, label_grid


class MissDiagnosisTests(unittest.TestCase):
    def test_partition_counts_are_disjoint_and_complete(self):
        errors=np.array([.5, 2., 3., 0.])
        pred=np.tile([-.1,.1],(4,1))
        result=miss_partition(errors,pred,np.zeros(4),np.ones(4),
                              np.array([True,True,False,True]),np.ones(4,bool))
        self.assertEqual(result["predicted_interval_misses"],3)
        self.assertEqual(result["missed_but_inside_observed_neighbor_hull"],1)
        self.assertEqual(result["missed_and_outside_observed_neighbor_hull"],1)
        self.assertEqual(result["missed_without_required_neighbor_evidence"],1)

    def test_brackets_do_not_pretend_edges_have_two_sides(self):
        lo,hi,valid,left,right=bracket_bounds(np.array([7.,8.]),np.array([[-1.],[-2.]]),
                np.array([[1.],[2.]]),np.array([6.,7.5,9.]),np.zeros(3,int))
        np.testing.assert_array_equal(valid,[False,True,False])
        self.assertEqual(lo[1],-2.)
        self.assertEqual(hi[1],2.)

    def test_grid_retains_duplicate_appearance_extremes(self):
        stream=dict(distance_m=np.ones(12)*7.,speed_mps=np.ones(12),
                    appearance=np.array([str(i) for i in range(12)]),image_action=np.arange(12)/10.,
                    q_action=np.zeros(12),current_barrier=np.ones(12),image_next_state=np.ones((12,2)))
        duplicate={k:np.concatenate((v,v[:1])) for k,v in stream.items()}
        duplicate["image_action"][-1]=-2.
        _,_,(low,high),by=label_grid(duplicate)
        self.assertEqual(low[0,0],-2.)
        self.assertEqual(high[0,0],1.1)
        self.assertEqual(by["0"][0][0,0],-2.)


if __name__ == "__main__":
    unittest.main()
