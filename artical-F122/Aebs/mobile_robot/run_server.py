"""Detached, finite experiment stages with persistent status and exit codes.

Run from the isolated server checkout. Never declares a certificate passed
from a process exit status: read the separate certificate.json.
"""
import argparse,datetime,json,os,subprocess,sys,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('job',choices=['visual','controller','audit'])
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2];os.chdir(root)
    work=root.parent/'work';work.mkdir(exist_ok=True)
    status_path=work/(args.job+'_status.json')
    if status_path.exists():raise FileExistsError('Choose a new job directory rather than overwrite job status')
    results=Path('results');prefix='mobile_robot_20261108_'
    baseline=results/(prefix+'ppo_r1');observer=results/(prefix+'perception_r1')
    dino=['--dino-repo','/root/work_based_on_spvc/external/dinov2',
          '--dino-weights','/root/work_based_on_spvc/external/dinov2_weights/dinov2_vits14_pretrain.pth']
    if args.job=='visual':
        refined=results/(prefix+'server_perception_r2')
        stages=[('image_r1','image_rollout',['--actor',str(baseline/'actor.pt'),'--observer',str(observer/'observer.pt'),
                    '--output-dir',str(results/(prefix+'server_image_r1')),'--episodes','24',*dino]),
                ('perception_r2','train_perception',['--output-dir',str(refined),'--resume',str(observer/'observer.pt'),
                    '--reuse-features',str(work/'features_portable.pt'),'--epochs','200','--learning-rate','.0003',*dino]),
                ('image_r2','image_rollout',['--actor',str(baseline/'actor.pt'),'--observer',str(refined/'observer.pt'),
                    '--output-dir',str(results/(prefix+'server_image_r2')),'--episodes','24',*dino])]
    elif args.job=='controller':
        actor=results/(prefix+'server_ppo_r2');sbc=results/(prefix+'server_sbc_r4')
        stages=[('ppo_r2','train_ppo',['--output-dir',str(actor),'--steps','500000','--radius','.06','--seed','8']),
                ('sbc_r4','train_sbc',['--actor',str(actor/'actor.pt'),'--output-dir',str(sbc),
                    '--radius','.06','--epochs','15000','--hidden','64','32']),
                ('audit_r4','certify',['--actor',str(actor/'actor.pt'),'--sbc',str(sbc/'sbc.pt'),
                    '--output-dir',str(results/(prefix+'server_audit_r4')),'--radius','.06','--depth','8'])]
    else:
        stages=[('audit_r3','certify',['--actor',str(baseline/'actor.pt'),
                    '--sbc',str(results/(prefix+'sbc_r3')/'sbc.pt'),
                    '--output-dir',str(results/(prefix+'server_audit_r3')),'--radius','.03','--depth','8','--active-level','1'])]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    status=dict(job=args.job,pid=os.getpid(),device='cpu',root=str(root),stages=[],state='running',
        started=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Both GPUs occupied; CUDA initialization failed with OOM')
    def save():
        temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(status_path)
    save()
    try:
        for name,module,options in stages:
            command=[sys.executable,'-m','Aebs.mobile_robot.'+module,*options]
            entry=dict(name=name,command=command,state='running',log=str(work/(name+'.log')))
            status['stages'].append(entry);save()
            with open(entry['log'],'w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root)
                entry['pid']=child.pid;save();code=child.wait()
            entry.update(exit_code=code,state='finished' if code==0 else 'failed');save()
            if code:raise RuntimeError('Stage failed: '+name)
        status['state']='finished'
    except Exception as exc:
        status['state']='failed';status['error']=str(exc);traceback.print_exc()
    status['finished']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    if status['state']=='failed':sys.exit(1)


if __name__=='__main__':main()
