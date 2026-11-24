"""Check native full-latent proxy action intervals against actual RGB actions.

Uses unchanged native network and its existing backward LiRPA option. Every
comparison uses the SAME first-frame physical state; no new teacher policy.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from .proxy_architecture_compare import sha,write
from .spvt_native_checkpoint_study import load_native


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rgb-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstream',type=Path,default=Path('/root/SPVT'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    sys.path[:0]=[str(a.upstream),str(a.upstream/'spvt')]
    from model import model as module
    from spvt.utils import create_bounded_module
    from auto_LiRPA import BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    import auto_LiRPA
    rgbprotocol=json.loads((a.rgb_root/'protocol.json').read_text())
    selected=rgbprotocol['native_checkpoint_selection']['selected']
    paths=dict(spvt_selected=Path(selected['path']),pid_initializer=a.upstream/'model/pretrained_ckpts/pid_controller.pth')
    protocol=dict(rgb_protocol_sha256=sha(a.rgb_root/'protocol.json'),
        methods={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        sources={str(p):sha(p) for p in [Path(__file__),a.upstream/'spvt/utils.py',a.upstream/'model/model.py']},
        auto_lirpa_file=str(Path(auto_LiRPA.__file__).resolve()),bound_method='backward',
        latent='entire [-.8,.8]^10; exact physical state, not a sampled latent approximation',
        measurement='test first camera frame action versus full latent-proxy action interval at same true state',
        scope='numerical action-inclusion diagnostic on target rendering domain; no baseline source changes',
        tolerance_degrees=1e-4,device=a.device,batch=a.batch)
    write(a.output/'protocol.json',protocol);summary={}
    reference=None
    for name,path in paths.items():
        data_path=a.rgb_root/(name+'__test.npz')
        with np.load(data_path,allow_pickle=False) as data:
            states=data['states_cm_degrees'][:,0].copy();observed=data['actions_normalized'][:,0].copy()
        if reference is None:reference=states
        else:np.testing.assert_array_equal(reference,states)
        model=load_native(module,a.upstream,path,a.device)
        cfg=argparse.Namespace(device=a.device)
        bounded=create_bounded_module(cfg,model)
        bounded.eval()
        lowers=[];uppers=[];nominal=[]
        for first in range(0,len(states),a.batch):
            x=torch.tensor(states[first:first+a.batch],device=a.device);count=len(x)
            if count<a.batch:x=torch.cat([x,x[-1:].repeat(a.batch-count,1)])
            z=torch.zeros(len(x),10,device=a.device)
            pert=PerturbationLpNorm(norm=np.inf,x_L=torch.full_like(z,-.8),x_U=torch.full_like(z,.8))
            zz=BoundedTensor(z,pert)
            with torch.no_grad():
                bounded(zz,x)
                lo,hi=bounded.compute_bounds(x=(zz,x),method='backward')
                middle=model(z,x)
            lowers.append(lo[:count].flatten().cpu().numpy());uppers.append(hi[:count].flatten().cpu().numpy())
            nominal.append(middle[:count].flatten().cpu().numpy())
            if first%128==0:print(json.dumps(dict(arm=name,completed=first+count,total=len(states))),flush=True)
        lower=np.concatenate(lowers)*30;upper=np.concatenate(uppers)*30;actual=observed*30;middle=np.concatenate(nominal)*30
        assert np.isfinite(lower).all() and np.isfinite(upper).all() and (lower<=upper).all()
        assert (lower<=middle+1e-4).all() and (upper>=middle-1e-4).all()
        distance=np.maximum(np.maximum(lower-actual,actual-upper),0)
        np.savez_compressed(a.output/(name+'.npz'),states_cm_degrees=states,actual_action_degrees=actual,
            proxy_action_lower_degrees=lower,proxy_action_upper_degrees=upper,nominal_action_degrees=middle,
            distance_outside_interval_degrees=distance)
        summary[name]=dict(n=len(states),contains_actual=int((distance<=1e-4).sum()),
            excludes_actual=int((distance>1e-4).sum()),mean_proxy_interval_width_degrees=float((upper-lower).mean()),
            mean_outside_distance_degrees=float(distance.mean()),p95_outside_distance_degrees=float(np.quantile(distance,.95)),
            first_frame_records_sha256=sha(data_path))
        print(json.dumps(summary[name]),flush=True)
    write(a.output/'summary.json',summary)


if __name__=='__main__':main()
