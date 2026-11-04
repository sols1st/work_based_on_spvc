import tempfile
import unittest
from pathlib import Path
import numpy as np
from Aebs.vision_sls_parking.core import INITIAL_STATES,GOAL_LOW,GOAL_HIGH,ACTION_HIGH,dynamics,nominal_observation,constraints
from Aebs.vision_sls_parking.planning_feedback import ReferenceBank
from Aebs.vision_sls_parking.interval_backup import step,safe,backup
from Aebs.vision_sls_parking.backup_governor import BackupGovernor,canonical_action,initial_belief,radius_upper
from Aebs.vision_sls_parking.backup_sbc import value,initial_cover
from Aebs.vision_sls_parking.perception_error import radius


class BackupSBCTests(unittest.TestCase):
    def test_interval_rk4_encloses_interior_and_vertices(self):
        rng=np.random.default_rng(19)
        for _ in range(30):
            center=rng.uniform([-3,-1.5,-3,-.7],[.3,1.6,3,1.7]);half=rng.uniform(0,.1,4)
            action=rng.uniform(-ACTION_HIGH,ACTION_HIGH)
            low,high=step(((center-half)[None],(center+half)[None]),action)
            for x in [center-half,center+half,*rng.uniform(center-half,center+half,(20,4))]:
                nxt=dynamics(x,action)
                self.assertTrue(((nxt>=low[0])&(nxt<=high[0])).all())

    def test_full_initial_union_has_outward_verified_backup(self):
        report,data=initial_cover(.02)
        self.assertTrue(report['passed'])
        self.assertTrue(all(safe((lo,hi)) for lo,hi in zip(data['low'],data['high'])))
        with self.assertRaises(ValueError):
            backup((data['low'][0],data['high'][0]),31)

    def test_action_normalization_fixed_point(self):
        for u in np.random.default_rng(7).uniform(-ACTION_HIGH,ACTION_HIGH,(100,2)):
            physical=canonical_action(u)
            np.testing.assert_array_equal(physical,np.clip(physical/ACTION_HIGH,-1,1)*ACTION_HIGH)

    def test_cp_radius_interval_dominates_interior(self):
        import itertools
        model=dict(low=[-3.5,-2.,-3.2],high=[.5,1.9,3.2],coefficients=np.random.default_rng(3).uniform(.1,2,(8,3)).tolist(),q=1.1)
        lo=np.array([-2.,-.3,-.2]);hi=lo+.1
        vertices=np.array([np.where(bits,hi,lo) for bits in itertools.product([False,True],repeat=3)])
        upper=radius_upper(vertices,model)
        interior=np.random.default_rng(4).uniform(lo,hi,(300,3))
        self.assertTrue((radius(interior,model)<=upper).all())

    def test_startup_hypothesis_contains_truth_under_its_error_box(self):
        model=dict(schema='parking_startup_group_cp_v1',initial_bound=.005,
                   bias=np.zeros((15,3)).tolist(),radius=np.full((15,3),.01).tolist())
        rng=np.random.default_rng(81)
        for j in range(15):
            for _ in range(8):
                x=INITIAL_STATES[j]+rng.uniform(-.005,.005,4)
                y=nominal_observation(x)+rng.uniform(-.01,.01,3)
                ids,_=initial_belief(y,(GOAL_LOW[j,:2]+GOAL_HIGH[j,:2])/2,model,.005)
                self.assertIn(j,ids)

    def test_analytic_sbc_invariant_with_adversarial_later_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bank.npz'
            np.savez(path,states=np.repeat(INITIAL_STATES[:,None,:],31,axis=1),actions=np.zeros((15,30,2)),
                     feedback=np.zeros((15,30,2,4)),observer=np.zeros((15,31,4,3)),goals=(GOAL_LOW[:,:2]+GOAL_HIGH[:,:2])/2)
            bank=ReferenceBank(path)
            class Policy:
                def predict(self,obs,deterministic=True):return np.ones(2),None
            model=dict(schema='parking_startup_group_cp_v1',initial_bound=.005,
                       bias=np.zeros((15,3)).tolist(),radius=np.full((15,3),.01).tolist())
            for j in [0,4,8]:
                governor=BackupGovernor(bank,Policy(),{},initial_jitter=.005,startup_model=model)
                x=INITIAL_STATES[j].copy();governor.reset(nominal_observation(x),bank.goals[j])
                self.assertEqual(value(x,governor),0.)
                for t in range(30):
                    u=governor.action();x=dynamics(x,u)
                    governor.receive(np.array([1e300,-1e300,1e300]) if t%3==0 else np.full(3,(-1)**t*10000.))
                    self.assertEqual(value(x,governor),0.)
                    self.assertFalse(constraints(x)['collision']);self.assertTrue(constraints(x)['in_bounds'])
                self.assertEqual(value([100,100,0,0],governor),20.)
                self.assertGreater(governor.invalid_measurements,0)

    def test_constructive_neural_sbc_separation_and_drift(self):
        import torch
        from types import SimpleNamespace
        from Aebs.vision_sls_parking.planned_sbc import PlannedBarrier
        from Aebs.vision_sls_parking.geometric_neural_sbc import GeometricNeuralSBC,relu_max
        from Aebs.vision_sls_parking.core import OBSTACLES,STATE_LOW,STATE_HIGH
        bank=SimpleNamespace(states=np.repeat(INITIAL_STATES[:,None,:],31,axis=1))
        model=GeometricNeuralSBC(PlannedBarrier(bank))
        # Large independent observer values exercise the bounded neural term.
        x=torch.tensor(INITIAL_STATES,dtype=torch.float64)
        state=torch.cat([x,torch.randn(15,4,dtype=torch.float64)*1e4],dim=1)
        refs=torch.arange(15);times=torch.zeros(15,dtype=torch.long)
        self.assertTrue((model.geometric(state)==0).all())
        current=model(state,refs,times)
        self.assertTrue((current<=.305+1e-14).all())
        successor=state.clone();successor[:,4:]*=-10
        following=model(successor,refs,times+1)
        self.assertTrue((following-current<=-.005+1e-14).all())
        unsafe=[]
        for k in range(4):
            a=INITIAL_STATES[0].copy();a[k]=STATE_LOW[k]-.001-1e-8;unsafe.append(a)
            a=INITIAL_STATES[0].copy();a[k]=STATE_HIGH[k]+.001+1e-8;unsafe.append(a)
        for px,py,r in OBSTACLES:
            for angle in np.linspace(0,2*np.pi,101):
                unsafe.append([px+np.sqrt(r*r+.0001)*np.cos(angle),py+np.sqrt(r*r+.0001)*np.sin(angle),0,0])
        self.assertTrue((model.geometric(torch.tensor(np.asarray(unsafe)))>=20-1e-9).all())
        values=torch.randn(300,32,dtype=torch.float64)*100
        torch.testing.assert_close(relu_max(values),values.max(-1).values,rtol=0,atol=1e-11)


if __name__=='__main__':unittest.main()
