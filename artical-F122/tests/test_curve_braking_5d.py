import itertools
import numpy as np
import torch
from Aebs.curve_braking import plant
from Aebs.curve_braking.models import Barrier,Actor,Proxy
from Aebs.curve_braking.verify import verify_checkpoint

def test_certificate_catches_excessive_distance_error_and_covers_nonzero_sublevel(tmp_path):
    b=Barrier().double();path=tmp_path/'barrier.pt'
    # An untrained candidate violates the initial bound; the checker must reject it.
    torch.save(b.state_dict(),path)
    assert not verify_checkpoint(path,.15)['passed']
    with torch.no_grad(): b.free.fill_(-3.)
    torch.save(b.state_dict(),path)
    assert verify_checkpoint(path,.15)['passed']
    assert not verify_checkpoint(path,.69)['passed']
    r=np.random.default_rng(124);x=r.uniform([-.65,-.19,1.92,0,-.07],[.65,.19,6.15,2.6,.07],(40000,5))
    val=lambda z:b(torch.tensor(z,dtype=torch.float64)).detach().numpy()
    old=val(x);active=old<20;x=x[active];old=old[active]
    assert (old>0).sum()>100
    for k,brake,steer in itertools.product(plant.KAPPAS,[1.2,3.5],[-.16,.16]):
        u=np.broadcast_to([-brake,steer],(len(x),2))
        nxt=plant.step(x,k,u)
        assert np.max(val(nxt)-old)<1e-10

def test_frenet_corner_clearance_conservative_for_both_bends():
    r=np.random.default_rng(19)
    x=r.uniform([-.65,-.19,1.92,0,-.07],[.65,.19,6.15,2.6,.07],(500,5))
    for k in plant.KAPPAS:
        lat,gap=plant.margins(x,k)
        for length,width in itertools.product([-1.6,1.6],[-.7,.7]):
            X=x[:,0]+length*np.sin(x[:,1])+width*np.cos(x[:,1])
            Y=length*np.cos(x[:,1])-width*np.sin(x[:,1])
            e=(1-np.hypot(1-k*X,k*Y))/k
            station=np.arctan2(k*Y,1-k*X)/k
            assert np.all(plant.ROAD_HALF-np.abs(e)>=lat)
            assert np.all(x[:,2]-station>=gap)

def test_nonlinear_coordinates_and_actuator_contract():
    torch.manual_seed(6);m=Proxy('neural',5)
    for net in m.nets:torch.nn.init.normal_(net[2].weight,std=.2)
    x=torch.randn(20,3);torch.testing.assert_close(m.decode(m.nominal(x)),x)
    near=x.clone();near[:,2]=1.75
    u=Actor()(near,torch.rand(20),torch.zeros(20),torch.full((20,),.02))
    assert (u[:,0]<=-1.2).all() and (u[:,1].abs()<=.16).all()
