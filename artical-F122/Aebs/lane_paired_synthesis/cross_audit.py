"""Exchange the frozen actor and B on one common centered physical observation box."""
import argparse
from pathlib import Path
import torch

from ..lane_keeping.learned_sbc import load_sbc_checkpoint,sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json
from .audit_supplement import audit
from .train import BASE,ARMS,read


def run(root,arm):
    p=root/'common_box_cross'
    cutoff=root/'fixed_corrected/uniform/final_contract.json'
    source=root/'protocol.json'
    bias=root/'bias.json'
    if arm=='register':
        p.mkdir(exist_ok=False)
        given=read(cutoff)
        assert given['centered'] and given['action'] is None
        nodes={'fixed':BASE,**{n:root/n for n in ARMS}}
        write_json(p/'protocol.json',dict(source=sha256(Path(__file__)),comparison_source=sha256(source),
            cutoff_source=sha256(cutoff),bias=sha256(bias),models={n:{m:sha256(v/(m+'.pt')) for m in ('actor','sbc')}
                for n,v in nodes.items()}, box=given['width'],margins=[0.,.1], curvatures=[0.,-.01,.01,-.02,.02],
            pairs='old-old reference; each new actor with old B, old actor with each new B, and each own new pair',
            status='supplement after seeing own final uniform box result; fixed known-curvature correction for all pairs',
            scope='conditional continuous-domain certificate only; recombination carries no coverage claim'))
        return
    torch.set_num_threads(1)
    spec=read(p/'protocol.json')
    assert spec['cutoff_source']==sha256(cutoff) and spec['source']==sha256(Path(__file__))
    assert spec['bias']==sha256(bias) and spec['comparison_source']==sha256(source)
    pairs=[('fixed','fixed')] if arm=='fixed' else [(arm,'fixed'),('fixed',arm),(arm,arm)]
    cfg={**read(source)['final_audit'],'margins':spec['margins']}
    for a,b in pairs:
        out=p/(a+'__'+b+'.json')
        if out.exists():raise FileExistsError(out)
        ap=(BASE if a=='fixed' else root/a)/'actor.pt'
        bp=(BASE if b=='fixed' else root/b)/'sbc.pt'
        assert sha256(ap)==spec['models'][a]['actor'] and sha256(bp)==spec['models'][b]['sbc']
        actor=load_actor(ap).double().requires_grad_(False)
        barrier=load_sbc_checkpoint(bp)[0].double().requires_grad_(False)
        row=audit(actor,barrier,dict(width=spec['box'],action=None),cfg,spec['curvatures'])
        write_json(out,dict(actor=a,sbc=b,protocol=sha256(p/'protocol.json'),audit=row))
        print(a,b,[r['status'] for r in row['rows']],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('register','fixed',*ARMS))
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    run(args.root,args.stage)
