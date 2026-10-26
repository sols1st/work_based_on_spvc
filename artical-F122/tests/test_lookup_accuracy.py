import unittest
import numpy as np
from Aebs.dino_latent.audit_image_lookup_accuracy import geometry,brute_force_check
from Aebs.dino_latent.experiment_sbc_risk_cp import lookup_alignment_gate
from tests.test_native_controllers import fixture


class LookupAccuracyTests(unittest.TestCase):
    def test_more_appearances_do_not_fill_missing_distances(self):
        a=geometry([5.,5.1,6.9,7.],np.array([6.]))
        b=geometry([5.,5.1,6.9,7.]*12,np.array([6.]))
        self.assertAlmostEqual(a["dense_query_error_m"]["max"],.9)
        self.assertEqual(a,b)

    def test_original_dense_grid_differs_from_block_split(self):
        dense=np.linspace(5,16,2001)
        sub=dense[(dense<6)|(dense>7)]
        q=np.array([6.5])
        self.assertLess(geometry(dense,q)["dense_query_error_m"]["max"],.003)
        self.assertGreater(geometry(sub,q)["dense_query_error_m"]["max"],.49)

    def test_searchsorted_agrees_with_independent_argmin(self):
        _,interface=fixture()
        result=brute_force_check(interface,np.array([5.,6.,7.,8.,10.,13.]),"a|b")
        self.assertEqual(result["nonminimal_distance_count"],0)
        self.assertEqual(result["selected_source_mismatch_count"],0)

    def test_strict_gate_checks_unvisited_holes_not_only_observed_frames(self):
        class Library:
            distances=np.array([5.,5.5,6.,8.,16.])
            pools={"appearance":np.arange(5)}
        self.assertFalse(lookup_alignment_gate(Library())["synchronized_image_eligible"])

    def test_dense_original_sampling_can_pass_alignment_gate(self):
        class Library:
            distances=np.linspace(5,16,2001)
            pools={"appearance":np.arange(2001)}
        self.assertTrue(lookup_alignment_gate(Library())["synchronized_image_eligible"])


if __name__=="__main__":unittest.main()
