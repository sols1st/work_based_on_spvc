"""Test the preregistered numerical variant after planning, preserving strict results."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--variant',choices=['convexified','regularized','conditioned'],default='convexified')
args=p.parse_args();root=args.root.resolve();py=root/'.venv/bin/python'
protocol=json.loads((root/'protocol.json').read_text());scene=protocol['scene']
suffix={'convexified':'_convexified_iter50000_outer80','regularized':'_convexified_iter50000_outer100_rf0p001','conditioned':'_convexified_iter100000_outer100_rf0p001_normalized'}[args.variant];out=root/'results'
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='8',MPLBACKEND='Agg',LIBGL_ALWAYS_SOFTWARE='1')
started=time.monotonic()
while not (out/('plan_status'+suffix+'.json')).exists():
    if time.monotonic()-started>2400:raise TimeoutError('Planner status missing')
    time.sleep(3)
status=json.loads((out/('plan_status'+suffix+'.json')).read_text())
def run(args,name):
    print('Starting '+name,flush=True)
    with (out/name).open('w') as f:subprocess.run([str(x) for x in args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
if status['plan_exported']:
    weights='dinov2_vitg14_pretrain.pth' if scene=='parking' else 'dinov2_vitl14_pretrain.pth'
    name='unicycle_of_perception_0.npz' if scene=='parking' else 'quad10d_perception_0.npz'
    run(['xvfb-run','-a','-s','-screen 0 1280x1024x24 +extension GLX +render -noreset',py,'-u',
      root/'run_visual_rollout.py','--source',root/'source','--weights',root/'assets'/weights,
      '--dinov2-source',root/'assets/dinov2','--output',out/('doc_cp'+suffix),
      '--rollouts','50','--seed',str(protocol['evaluation']['seed']),
      '--plan',out/('doc_cp_plan'+suffix)/'data'/name,'--state-cp-model',out/'doc_cp_model.json'],
      'doc_cp'+suffix+'.log')
else:print('Numerically adapted planner also failed; no closed-loop test.',flush=True)
run([py,root/'analyze_doc_cp.py','--root',root,'--variant',args.variant],'analysis'+suffix+'.log')
print('Complete numerical variant',flush=True)
