"""Symmetric fine-grid sensitivity analysis after the registered five-point audit."""
import argparse
from pathlib import Path
import torch

from ..lane_keeping.learned_sbc import load_sbc_checkpoint,sha256
from ..lane_keeping.policy import load_actor
from ..lane_keeping.robust_trajectory_cp import write_json
from .audit_supplement import audit
from .train import BASE,ARMS,read


def run(root,mode):
    dest=root/'margin_sensitivity'
    if mode=='register':
        dest.mkdir(exist_ok=False)
        inputs={}
        for arm in ARMS:
            for new in ['fixed_corrected','trained']:
                folder=root/arm if new=='trained' else root/new/arm
                inputs[new+'_'+arm]=sha256(folder/'final_contract.json')
        write_json(dest/'protocol.json',dict(source=sha256(Path(__file__)),root_protocol=sha256(root/'protocol.json'),
                    contracts=inputs,model='fixed_corrected and each resynthesized pair',
                    range=[.11,.12,.13,.14,.145,.15],
                    analysis='additional grid specified after registered coarse results; certified widths are linear angular amplitudes',
                    shared_config='all original verification settings incl 180 seconds per margin and system',
                    reject_larger='earlier true witness retained when expanding symmetrical extra steering range'))
        return
    spec=read(dest/'protocol.json')
    assert spec['source']==sha256(Path(__file__)) and spec['root_protocol']==sha256(root/'protocol.json')
    new,arm=mode.rsplit('_',1)
    folder=root/arm if new=='trained' else root/new/arm
    pair=root/arm if new=='trained' else BASE
    cp=read(folder/'final_contract.json')
    assert spec['contracts'][mode]==sha256(folder/'final_contract.json')
    assert sha256(pair/'actor.pt')==cp['actor_sha256'] and sha256(pair/'sbc.pt')==cp['sbc_sha256']
    fixed=folder/('audit.json' if new=='trained' else 'supplemental_audit.json')
    coarse=read(fixed)
    if new!='trained':coarse=coarse['all']
    torch.set_num_threads(1)
    a=load_actor(pair/'actor.pt').double().requires_grad_(False)
    b=load_sbc_checkpoint(pair/'sbc.pt')[0].double().requires_grad_(False)
    cfg=read(root/'protocol.json')['final_audit'].copy()
    found=[(r['normalized_margin'],str(fixed)) for r in coarse['rows']
           if r.get('five_curvatures',r.get('status'))=='witness']
    out=dest/(mode+'.json')
    if out.exists():raise FileExistsError(out)
    rows=[]
    for level in sorted(spec['range']):
        witness=next(((m,p) for m,p in found if m<=level),None)
        if witness is not None:
            row=dict(normalized_margin=level,steering_rad=level*.45,status='witness_by_inclusion',
                     source_margin=witness[0],source_file=witness[1])
        else:
            cfg['margins']=[level]
            candidate=audit(a,b,cp,cfg,[0.,-.01,.01,-.02,.02])
            row=candidate['rows'][0]
            if row['status']=='witness':found.append((level,str(out)))
        rows.append(row)
        print(mode,level,row['status'],flush=True)
    write_json(out,dict(arm=arm,pair=new,protocol=sha256(dest/'protocol.json'),coarse=sha256(fixed),rows=rows))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root',type=Path)
    cli.add_argument('mode',choices=['register']+[tag+'_'+arm for tag in ('trained','fixed_corrected') for arm in ARMS])
    args=cli.parse_args()
    run(args.root,args.mode)
