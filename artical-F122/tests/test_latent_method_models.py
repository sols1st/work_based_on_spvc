import unittest
import numpy as np
import torch

from Aebs.dino_latent.latent_method_models import (
    make_model, predict, split_three, paired_cluster_bootstrap, GaussianRSSM, temporal_offsets,
)
from Aebs.dino_latent.benchmark_static_latents import FeatureAutoencoder
from Aebs.dino_latent.audit_static_decomposition import decompose


class LatentMethodModelsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)

    def test_three_splits_disjoint_and_complete(self):
        parts=split_three(459)
        self.assertEqual(tuple(map(len,parts)),(321,69,69))
        self.assertEqual(sorted(np.concatenate(parts).tolist()),list(range(459)))
        for a,b in ((0,1),(0,2),(1,2)): self.assertFalse(set(parts[a])&set(parts[b]))
        for a,b in zip(parts,split_three(459)): np.testing.assert_equal(a,b)

    def test_all_models_roll_and_backpropagate(self):
        for kind in ("mlp","mlp_matched","transformer","no_speed","task_aware","gru","lifted_linear","rssm"):
            model=make_model(kind,8)
            target=torch.randn(3,5,8)
            value,aux=predict(model,torch.randn(3,3,8),torch.ones(3,3),torch.randn(3,7),5,target)
            self.assertEqual(value.shape,(3,5,8))
            loss=(value-target).square().mean()+aux
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_new_models_future_action_causality(self):
        z,v,a=torch.randn(3,3,8),torch.ones(3,3),torch.randn(3,7)
        changed=a.clone(); changed[:,-1]+=2
        for kind in ("gru","lifted_linear","rssm"):
            model=make_model(kind,8).eval()
            if kind=="gru": torch.nn.init.normal_(model.head.weight,std=.1)
            x,_=predict(model,z,v,a,5); y,_=predict(model,z,v,changed,5)
            torch.testing.assert_close(x[:,:4],y[:,:4],rtol=0,atol=0)
            self.assertFalse(torch.allclose(x[:,-1],y[:,-1]))

    def test_rssm_predictions_do_not_consume_future_targets(self):
        model=GaussianRSSM(dimension=8).eval()
        z,v,a=torch.randn(2,3,8),torch.ones(2,3),torch.zeros(2,6)
        x,_=predict(model,z,v,a,4,torch.zeros(2,4,8))
        y,_=predict(model,z,v,a,4,torch.full((2,4,8),100.))
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(model.kl(torch.zeros(2,4),torch.ones(2,4),torch.zeros(2,4),torch.ones(2,4))))

    def test_cluster_ci_sign_and_equal_trajectory_weight(self):
        groups=np.array([1,1,1,2])
        result=paired_cluster_bootstrap(np.array([0.,0.,0.,2.]),np.ones(4),groups)
        self.assertEqual(result["mean_paired_delta"],0.)
        self.assertEqual(result["trajectory_count"],2)
        self.assertEqual(result["lower_error_trajectory_fraction"],.5)
        good=paired_cluster_bootstrap(np.zeros(4),np.ones(4),groups)
        self.assertEqual(good["ci95"],[-1.,-1.])

    def test_bad_bootstrap_rejected(self):
        with self.assertRaises(ValueError): paired_cluster_bootstrap([0],[1],[0])
        with self.assertRaises(ValueError): paired_cluster_bootstrap([np.nan,0],[1,1],[0,1])
        with self.assertRaises(ValueError): make_model("unknown")

    def test_approximate_parameter_match_is_within_two_percent(self):
        n=lambda kind:sum(p.numel() for p in make_model(kind).parameters())
        self.assertLess(abs(n("mlp_matched")/n("transformer")-1),.02)

    def test_action_offset_length_matches_multistep_rollout(self):
        for horizon in (3,10):
            c,t,a=temporal_offsets(3,horizon)
            self.assertEqual((len(c),len(t),len(a)),(3,horizon,3+horizon-1))
            self.assertEqual(a[-1],horizon-1)
            z=torch.randn(2,len(c),8)
            model=make_model("mlp",8)
            out,_=predict(model,z,torch.ones(2,len(c)),torch.zeros(2,len(a)),horizon)
            self.assertEqual(out.shape,(2,horizon,8))

    def test_autoencoders_mean_readout_and_kl(self):
        x=torch.randn(4,384)
        for variational in (False,True):
            codec=FeatureAutoencoder(variational)
            reconstruction,kl=codec(x)
            self.assertEqual(reconstruction.shape,x.shape)
            self.assertTrue(torch.isfinite(kl))
            (reconstruction.square().mean()+.001*kl).backward()
            self.assertGreater(float(codec.encoder[0].weight.grad.abs().sum()),0)
            codec.eval()
            a,ka=codec(x); b,kb=codec(x)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if not variational: self.assertEqual(float(ka),0.)
        for parameter in codec.encoder.parameters(): parameter.data.zero_()
        _,kl=codec(x)
        self.assertEqual(float(kl),0.)

    def test_group_decomposition_identity(self):
        image=np.array([[0.,2.],[2.,0.],[5.,5.],[7.,7.]])
        proxy=np.array([[0.,0.],[0.,0.],[7.,7.],[7.,7.]])
        r=decompose(image,proxy,[0,0,1,1])
        self.assertLess(r["identity_abs_error"],1e-10)
        self.assertEqual(r["pair_mse"],2.)
        self.assertEqual(r["same_distance_appearance_mse"],1.)

    def test_task_objective_control_has_identical_initial_architecture(self):
        torch.manual_seed(7); a=make_model("transformer")
        torch.manual_seed(7); b=make_model("task_aware")
        self.assertEqual(a.state_dict().keys(),b.state_dict().keys())
        for k,v in a.state_dict().items(): torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)


if __name__=="__main__": unittest.main()
