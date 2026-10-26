"""Continue the registered experiment after the already-dispatched calibration."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();py=root/'.venv/bin/python'
protocol=json.loads((root/'protocol.json').read_text());scene=protocol['scene']
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         NUMBA_NUM_THREADS='8',MPLBACKEND='Agg',LIBGL_ALWAYS_SOFTWARE='1')
def wait_file(path,log):
    started=time.monotonic()
    while not path.exists():
        if log.exists() and 'Traceback (most recent call last)' in log.read_text(errors='replace'):
            raise RuntimeError('Predecessor failed: '+str(log))
        if time.monotonic()-started>1800:raise TimeoutError(str(path))
        time.sleep(3)
def run(args,log):
    print('Running '+str(args[1]),flush=True)
    with log.open('w') as f:subprocess.run([str(x) for x in args],env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,check=True)
wait_file(root/'results/calibration/summary.json',root/'results/calibration.log')
run([py,root/'calibrate_state_cp.py','--root',root],root/'results/calibrate_state_cp.log')
try:
    run([py,root/'plan_state_cp.py','--root',root],root/'results/plan_state_cp.log')
except subprocess.CalledProcessError:
    audit=json.loads((root/'results/qp_audit.json').read_text())
    assert any(r.get('solver_status')=='problem non convex' for r in audit['records']), 'A failure other than nonconvex QP requires separate diagnosis'
    failure=root/'results/strict_replacement_failure';failure.mkdir(exist_ok=True)
    shutil.copy2(root/'results/qp_audit.json',failure/'qp_audit.json')
    shutil.copy2(root/'results/plan_state_cp.log',failure/'plan_state_cp.log')
    adaptation={'reason':'OSQP rejected indefinite local quadratic objective',
       'method':'Symmetric eigendecomposition; project eigenvalues to at least 1e-6 only when minimum eigenvalue < -1e-7',
       'changes':'local SCP quadratic surrogate only; predictor, CP offset, original objective expression, constraints, dynamics, horizon and test seeds unchanged',
       'decided_before_state_cp_test':True,'strict_original_solver_succeeded':False}
    (root/'results/solver_adaptation.json').write_text(json.dumps(adaptation,indent=2)+'\n')
    run([py,root/'plan_state_cp.py','--root',root,'--convexify'],root/'results/plan_convexified.log')
weights='dinov2_vitg14_pretrain.pth' if scene=='parking' else 'dinov2_vitl14_pretrain.pth'
filename='unicycle_of_perception_0.npz' if scene=='parking' else 'quad10d_perception_0.npz'
run(['xvfb-run','-a','-s','-screen 0 1280x1024x24 +extension GLX +render -noreset',
     py,'-u',root/'run_visual_rollout.py','--source',root/'source',
     '--weights',root/'assets'/weights,'--dinov2-source',root/'assets/dinov2',
     '--output',root/'results/state_cp','--rollouts','50','--seed',str(protocol['evaluation']['seed']),
     '--plan',root/'results/state_cp_plan/data'/filename,
     '--state-cp-model',root/'results/state_cp_model.json'],root/'results/state_cp.log')
wait_file(root/'results/baseline/summary.json',root/'results/baseline.log')
run([py,root/'compare_state_cp.py','--root',root],root/'results/comparison.log')
print('Complete: '+str(root/'results/state_cp_comparison.json'),flush=True)
