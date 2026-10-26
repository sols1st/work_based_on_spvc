import copy
import importlib.util
import unittest
import numpy as np
from Aebs.vision_sls_parking.adapt_projection import (
    make_split_plan, cell_ids, historical_replay_indices, choose_epoch, train_candidate,
)


class AdaptationPlanTest(unittest.TestCase):
    def test_spatial_splits_are_disjoint_and_counts_fixed(self):
        x, cells, splits = make_split_plan()
        self.assertEqual(x.shape, (1024, 4))
        self.assertEqual({s: int((splits == s).sum()) for s in np.unique(splits)},
                         dict(train=640, validation=192, test=192))
        sets = {s: set(cells[splits == s]) for s in np.unique(splits)}
        self.assertEqual([len(sets[s]) for s in ['train','validation','test']], [40,12,12])
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        np.testing.assert_equal(cells, cell_ids(x))
        self.assertTrue((x[:, 3] == 0).all())

    def test_plan_repeatable(self):
        first, second = make_split_plan(), make_split_plan()
        for a,b in zip(first,second): np.testing.assert_equal(a,b)
        with self.assertRaises(ValueError): make_split_plan(0)

    def test_historical_replay_never_in_validation_or_test_cells(self):
        x,cells,splits=make_split_plan(3)
        source=np.vstack([x,[5,0,0,0],[-5,0,0,0]])
        ids=historical_replay_indices(source,cells,splits)
        self.assertTrue(np.isin(cell_ids(source[ids]),np.unique(cells[splits=='train'])).all())
        self.assertFalse(np.isin(cell_ids(source[ids]),np.unique(cells[splits=='test'])).any())

    def test_out_of_domain_not_clipped_into_training_cell(self):
        np.testing.assert_equal(cell_ids([[-3.001,0,0,0],[.001,0,0,0],[-1,2,0,0]]),[-1,-1,-1])
        with self.assertRaises(ValueError): cell_ids([[np.nan]*4])

    def test_validation_selection_includes_original_epoch_zero(self):
        self.assertEqual(choose_epoch([.1,.2,.15]),0)
        self.assertEqual(choose_epoch([.3,.1,.1]),1)
        with self.assertRaises(ValueError): choose_epoch([np.nan])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch unavailable in local light environment')
class AdaptationTrainingTest(unittest.TestCase):
    def test_only_projection_updates_and_baseline_retained(self):
        import torch
        from torch import nn
        class TinyProjection(nn.Module):
            def __init__(self):
                super().__init__()
                self.net=nn.Linear(5,3)
                self.C=nn.Parameter(torch.randn(3,4))
            def forward(self,x): return self.net(x)
        torch.set_num_threads(1); torch.manual_seed(7)
        model=TinyProjection().eval().requires_grad_(False)
        saved=copy.deepcopy(model.state_dict())
        x=torch.randn(64,5); w=torch.randn(5,3); y=x@w
        new,info=train_candidate(model,x,y,x,y,x,y,epochs=15,learning_rate=.03)
        self.assertGreater(info['best_epoch'],0)
        self.assertLess(min(info['validation_mse_by_epoch']),info['validation_mse_by_epoch'][0])
        self.assertFalse(info['test_used_for_selection'])
        self.assertTrue(info['C_bitwise_unchanged'])
        for k,v in model.state_dict().items(): self.assertTrue(torch.equal(v,saved[k]))
        self.assertTrue(torch.equal(new.C,saved['C']))
        self.assertFalse(any(p.requires_grad for p in new.parameters()))


if __name__ == '__main__': unittest.main()
