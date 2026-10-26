"""Run registered document-method stages; no adaptive model or solver fallback."""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
root=p.parse_args().root.resolve();py=root/'.venv/bin/python'
protocol=json.loads((root/'protocol.json').read_text());scene=protocol['scene']
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
 NUMBA_NUM_THREADS='8',MPLBACKEND='Agg',LIBGL_ALWAYS_SOFTWARE='1')
def wait(path,log):
    start=time.monotonic()
    while not path.exists():
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):raise RuntimeError(str(log))
        if time.monotonic()-start>2400:raise TimeoutError(str(path))
        time.sleep(3)
def run(args,name,check=True):
    print('Starting '+name,flush=True)
    with (root/'results'/name).open('w') as f:
        return subprocess.run([str(x) for x in args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=check).returncode
wait(root/'results/doc_envelope.json',root/'results/fit.log')
wait(root/'results/calibration/summary.json',root/'results/calibration.log')
run([py,root/'calibrate_doc_cp.py','--root',root],'calibrate.log')
code=run([py,root/'plan_doc_cp.py','--root',root],'plan.log',False)
if code==0:
    weights='dinov2_vitg14_pretrain.pth' if scene=='parking' else 'dinov2_vitl14_pretrain.pth'
    name='unicycle_of_perception_0.npz' if scene=='parking' else 'quad10d_perception_0.npz'
    run(['xvfb-run','-a','-s','-screen 0 1280x1024x24 +extension GLX +render -noreset',py,'-u',
      root/'run_visual_rollout.py','--source',root/'source','--weights',root/'assets'/weights,
      '--dinov2-source',root/'assets/dinov2','--output',root/'results/doc_cp',
      '--rollouts','50','--seed',str(protocol['evaluation']['seed']),
      '--plan',root/'results/doc_cp_plan/data'/name,'--state-cp-model',root/'results/doc_cp_model.json'],'doc_cp.log')
else:print('Original planning failed; no invented fallback controller.',flush=True)
wait(root/'results/baseline/summary.json',root/'results/baseline.log')
run([py,root/'analyze_doc_cp.py','--root',root],'analysis.log')
print('Complete '+str(root/'results/doc_cp_comparison.json'),flush=True)
