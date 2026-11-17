"""Compare actual checkpoint tensors and collect independent audit evidence."""
import argparse
import json
from pathlib import Path
import torch
from .run import sha256


def tensors(obj,prefix=''):
    result={}
    if isinstance(obj,torch.Tensor): return {prefix:obj.detach().cpu()}
    if isinstance(obj,dict):
        for key,value in obj.items():result.update(tensors(value,prefix+'/'+str(key)))
    return result


def compare(before,after):
    old=torch.load(before,map_location='cpu',weights_only=False)
    new=torch.load(after,map_location='cpu',weights_only=False)
    # Strip SBC/policy packaging metadata, comparing the actual model values.
    old=tensors(old.get('state_dict',old));new=tensors(new.get('state_dict',new))
    if old.keys()!=new.keys():raise ValueError('checkpoint tensor names differ')
    delta={k:new[k].double()-v.double() for k,v in old.items()}
    return dict(changed=any(torch.any(d!=0).item() for d in delta.values()),
        l2=float(sum(d.square().sum() for d in delta.values()).sqrt()),
        max_abs=max(float(d.abs().max()) for d in delta.values()),
        old_sha256=sha256(before),new_sha256=sha256(after))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();out=a.run;protocol=json.loads((out/'protocol.json').read_text())
    source=Path(protocol['source']);package=out/'package';lane=protocol['task']=='lane'
    report=json.loads((out/'report.json').read_text())
    if report['status'] not in ['passed_numerical','final_not_certified']:raise ValueError('final stage not reached')
    actor='actor.pt' if lane else 'policy.pt';observer='patch_projection.pt' if lane else 'observation.pt'
    result=dict(task=protocol['task'],status=report['status'],actor=compare(source/actor,package/actor),
        barrier=compare(source/'sbc.pt',package/'sbc.pt'),frontend_identical=sha256(source/observer)==sha256(package/observer),
        cp=json.loads((out/'final_cp/report.json').read_text()),
        audit=json.loads((out/'final_audit'/('certificate.json' if lane else 'report.json')).read_text()),
        training_checkpoints='new random barrier lineage; old barrier used only for initial actor feedback',
        code_manifest_sha256=sha256(out/'source_manifest.json') if (out/'source_manifest.json').exists() else None)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['task','status','actor','barrier','frontend_identical']}))


if __name__=='__main__':main()
