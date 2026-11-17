import torch
from Aebs.lane_keeping.patch_perception import PatchProjection
from Aebs.lane_keeping.compare_proxy_train import make_pair,objective,audit_geometry


def test_matched_pair_starts_identically_and_both_receive_gradients():
    torch.manual_seed(1)
    visual=PatchProjection(4,4,2)
    saved={'feature_dim':4,'hidden':4,'grid':2,'projection':visual.state_dict()}
    pair=make_pair(saved,123)
    x=torch.rand(8,2)-.5;features=torch.randn(8,4,4)
    a,b=pair.values()
    torch.testing.assert_close(a(features),b(features),rtol=0,atol=0)
    torch.testing.assert_close(a.nominal(x),b.nominal(x),rtol=0,atol=0)
    torch.testing.assert_close(objective(a,features,x),objective(b,features,x),rtol=0,atol=0)
    for m in pair.values():
        loss=objective(m,features,x);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        assert m.raw_C.grad.abs().sum()>0
        assert m.visual.out.weight.grad.abs().sum()>0
    assert b.f[2].weight.grad.abs().sum()+b.g[2].weight.grad.abs().sum()>0
    assert audit_geometry(a)['best_affine_fit_max_abs']<1e-12
    assert audit_geometry(b)['inverse_roundtrip_max_abs']<1e-12


def test_shared_image_observer_adapter_preserves_linear_and_neural_semantics(tmp_path,monkeypatch):
    from torch import nn
    from Aebs.lane_keeping import compare_proxy_cp as cp
    from Aebs.lane_keeping.learned_sbc import sha256
    class Backbone(nn.Module):
        def forward_features(self,images):
            pixels=images.permute(0,2,3,1).reshape(len(images),4,3)
            return {'x_norm_patchtokens':torch.cat([pixels,torch.ones_like(pixels[:,:,:1])],2)}
    class FakeDino:
        def __init__(self,*args):self.backbone=Backbone()
        def preprocess(self,x):return x
    monkeypatch.setattr(cp,'FrozenDinoV2',FakeDino)
    weights=tmp_path/'weights';weights.write_bytes(b'test')
    original=PatchProjection(4,4,2)
    pair=make_pair(dict(feature_dim=4,hidden=4,grid=2,projection=original.state_dict()),12)
    images=torch.randint(0,256,(3,2,2,3),dtype=torch.uint8)
    for kind,m in pair.items():
        path=tmp_path/(kind+'.pt')
        torch.save(dict(format='lane_joint_observation_v1' if kind=='linear' else 'lane_nonlinear_observation_v1',
            feature_dim=4,hidden=4,grid=2,state_dict=m.state_dict(),C=m.matrix().detach(),max_condition=10.,
            dino_weights_sha256=sha256(weights),dino_model='test'),path)
        observer=cp.MatchedImageObserver(path,kind,'cpu',tmp_path,weights)
        decoded,latent=observer.forward_with_latent(images)
        features=Backbone().forward_features(images.float().permute(0,3,1,2)/255.)['x_norm_patchtokens']
        torch.testing.assert_close(latent,m(features))
        torch.testing.assert_close(decoded,m.decode(m(features)))
