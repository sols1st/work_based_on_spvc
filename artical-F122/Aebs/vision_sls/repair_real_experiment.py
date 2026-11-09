"""Rebuild explicit metre contracts and rerun frozen-controller experiments.

Preserves old packages. Static distance-bin scores are empirical envelopes,
not a demonstrated exchangeable-image or new-image trajectory guarantee.
"""
import argparse,copy,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from .run import sha256
from .conformal import validate_contract_units
from .deploy import VisionSLSController
from .rollout import rollout,initial_grid


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def grouped_behavior(policy,scale,library,size=20):
    """One fixed weather/color per run; nearest lookup stays in that group."""
    output={}
    for split in ['all','test']:
        rows=[]
        for weather,color in sorted(set(zip(library['weather'].tolist(),library['color'].tolist()))):
            mask=(library['weather']==weather)&(library['color']==color)
            if split=='test':mask&=library['split']=='test'
            if not mask.any():raise ValueError('empty appearance group')
            r=rollout(policy,scale,initial_grid(size),library['distance_m'][mask],library['observation'][mask])
            rows.append(dict(weather=str(weather),color=str(color),images=int(mask.sum()),episodes=r['episodes'],
                counts=r['counts'],lookup_max_error_m=r['lookup_max_error_m']))
        output[split]=rows
    return dict(mode='frozen-image nearest replay, each episode keeps weather/color fixed',
        new_online_images=False,groups=output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact-dir',type=Path,required=True);p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-depth',type=int,default=18)
    args=p.parse_args(argv)
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(2)
    source=args.artifact_dir;cfg=json.loads((source/'config.json').read_text());s=cfg['barrier_distance_scale_m']
    original=json.loads((source/'perception_contract.json').read_text())
    metrics=json.loads((source/'metrics.json').read_text())
    reports={};status=dict(state='running',device='cpu',artifact=str(source),results=reports)
    write(args.output_dir/'status.json',status)
    for name in ['rgb','gray','cgan']:
        if name=='rgb':
            q=original['calibration']['q_hat'];coverage=original['coverage'];source_stats=source/'metrics.json'
        else:
            source_stats=args.baseline_dir/('gray_dino.json' if name=='gray' else 'cgan_state_net.json')
            m=json.loads(source_stats.read_text());q=m['cp']['q_norm']
            coverage=dict(calibration=dict(samples=m['cp']['n'],mean_radius_m=q*s,fraction=m['cp']['cal_coverage']),
                          test=dict(samples=m['test']['n'],fraction=m['cp']['test_coverage']))
        contract=copy.deepcopy(original);contract.update(coefficients=[s]*4,envelope_units='m',score_units='normalized_distance',
            distance_scale_m=s,coverage=coverage,evaluation_frontend=name,
            evaluation_role='fixed actor/SBC sensitivity to empirical error radius',
            source_statistics_sha256=sha256(source_stats),
            statistical_status='distance-bin held-out empirical envelope; ordinary image-level exchangeability NOT established',
            independent_trajectory_calibration=False,formal_image_certificate=False)
        contract['fit']['note']='constant metre coefficients; radius = normalized q * distance scale'
        contract['calibration']['q_hat']=q;contract['coverage']['calibration']['mean_radius_m']=q*s
        contract['note_scope']='Controller artifact stays fixed. Gray/cGAN are radius sensitivity comparisons, not RGB encoder identities.'
        validate_contract_units(contract);path=args.output_dir/('contract_'+name+'.json');write(path,contract)
        command=[sys.executable,'-m','Aebs.vision_sls.verify_sbc_cells','--artifact-dir',str(source),
            '--barrier-dir',str(source),'--perception-contract',str(path),'--output-dir',str(args.output_dir/('tube_'+name)),
            '--max-depth',str(args.max_depth),'--max-cells','1000000']
        with (args.output_dir/('tube_'+name+'.log')).open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((args.output_dir/('tube_'+name)/'report.json').read_text())
        reports[name]=dict(radius_m=q*s,passed=result['continuous_state_numerical_interval_pass'],
            actual_drift_witnesses=result['actual_witnesses'],unresolved=result['unresolved_cells'],
            statistics_sha256=sha256(source_stats),contract_sha256=sha256(path))
        write(args.output_dir/'status.json',status);print(json.dumps({name:reports[name]}),flush=True)
    controller=VisionSLSController(source);library=dict(np.load(source/'observations.npz',allow_pickle=False))
    behavior=grouped_behavior(controller.policy,s,library)
    write(args.output_dir/'grouped_behavior.json',behavior)
    status.update(state='finished',safety_claim='No new-image probabilistic safety claim; see corrected tube results and empirical replay.')
    write(args.output_dir/'status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
