"""Execute preregistered paired frontend training and six image-trajectory audits."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
from .learned_sbc import sha256


def write(p,x):
    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(p)


def worker(root,index):
    protocol=json.loads((root/'protocol.json').read_text());history=[]
    for i,seed in enumerate(protocol['training_seeds']):
        kind='linear' if (i+index)%2==0 else 'neural'
        train=root/f'train_{seed}'
        while not (train/'report.json').exists():
            if (root/'training_status.json').exists():
                s=json.loads((root/'training_status.json').read_text())
                if s.get('returncode',0):raise RuntimeError('training failed; no completed pair')
            time.sleep(5)
        output=root/f'cp_{seed}_{kind}'
        cmd=[sys.executable,'-m','Aebs.lane_keeping.compare_proxy_cp','--kind',kind,
             '--observer',str(train/kind/'observation.pt'),'--actor',protocol['actor'],'--sbc',protocol['sbc'],
             '--dino-repo',protocol['dino_repo'],'--dino-weights',protocol['dino_weights'],
             '--output-dir',str(output),'--comparison-protocol',str(root/'protocol.json'),
             '--device',f'cuda:{index}','--batch-size','8','--calibration-seed',str(protocol['calibration_seed']),
             '--holdout-seed',str(protocol['holdout_seed'])]
        history.append(dict(seed=seed,kind=kind,state='running',command=cmd));write(root/f'worker_{index}.json',history)
        start=time.monotonic()
        with (root/f'cp_{seed}_{kind}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        history[-1].update(state='complete' if not r.returncode else 'error',returncode=r.returncode,seconds=time.monotonic()-start)
        write(root/f'worker_{index}.json',history);r.check_returncode()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,choices=[0,1]);a=p.parse_args();root=a.root
    if a.worker is not None:return worker(root,a.worker)
    if (root/'protocol.json').exists():raise FileExistsError('choose a fresh experiment protocol')
    features=json.loads((root/'feature_protocol.json').read_text())
    actor=Path('results/lane_robust_synthesis_20261116_v2/package/actor.pt');sbc=actor.with_name('sbc.pt')
    protocol=dict(format='lane_matched_linear_neural_proxy_v1',date='2026-11-17',
        training_seeds=[202611171,202611172,202611173],epochs=30,batch_size=64,optimizer='Adam',lr=1e-4,
        arms=['trainable linear Cx','invertible neural coupling h_phi(x)'],loss='latent L2 + 0.1 decoded-state MSE, identical in both arms',
        shared_visual_warm_start='results/lane_keeping_20261105/patch_perception_v2/patch_projection.pt',
        data='data/lane_keeping_v2/dataset.npz',data_rows=3600,train_rows=2160,validation_rows=540,static_calibration_rows=540,test_rows=360,
        fresh_feature_protocol_sha256=sha256(root/'feature_protocol.json'),feature_cache_sha256=features['cache_sha256'],
        actor=str(actor),sbc=str(sbc),actor_sha256=sha256(actor),sbc_sha256=sha256(sbc),
        backend='same frozen robust controller and neural SBC for all six runs; no backend retraining',
        dino_repo='../external/dinov2',dino_weights='../external/dinov2_weights/dinov2_vits14_pretrain.pth',
        calibration_seed=202611170101,holdout_seed=202611170102,calibration_episodes=199,holdout_episodes=100,horizon=80,alpha=.05,
        pairing='same initial-state/curvature draws across every model; each model has separate feedback trajectories and CP threshold',
        gpu_assignment='GPU 0: linear/neural/linear; GPU 1: neural/linear/neural across ordered seeds; both RTX4090, batch8',
        primary_metrics=['decoded static error','trajectory CP radius','held-out complete-trajectory coverage','unsafe episodes','common SBC verification pass'],
        secondary_metrics=['raw latent residual','affine-fit residual of learned proxy','parameters','training time'],
        selection='each checkpoint selected only by its own validation decoded MSE and C condition<=10; report all seeds, no test/CP cherry-picking',
        statistical_scope='three paired training seeds, not six independent image datasets; paired rollout streams are not pooled as independent episodes across models',
        limitations='warm-start adaptation; same optimization-update budget, not identical parameter or wall-clock budget; nominal renderer law only')
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for name in ['compare_proxy_run.py','compare_proxy_train.py','compare_proxy_cp.py','nonlinear_observation.py','joint_observation.py','patch_perception.py','trajectory_cp.py','core.py','renderer.py','policy.py','verify_learned.py','verify.py','learned_sbc.py']:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    protocol['source_sha256']={p.name:sha256(p) for p in snapshot.iterdir()}
    write(root/'protocol.json',protocol)
    command=json.loads((root/'training_command.json').read_text())+['--train-only']
    write(root/'training_status.json',dict(state='running',command=command))
    handles=[]
    f=(root/'training.log').open('w');handles.append(f)
    train=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'})
    workers=[]
    for index in [0,1]:
        f=(root/f'worker_{index}.log').open('w');handles.append(f)
        workers.append(subprocess.Popen([sys.executable,'-m','Aebs.lane_keeping.compare_proxy_run','--root',str(root),'--worker',str(index)],stdout=f,stderr=subprocess.STDOUT))
    code=train.wait();write(root/'training_status.json',dict(state='complete' if not code else 'error',returncode=code,command=command))
    codes=[w.wait() for w in workers]
    write(root/'execution_status.json',dict(training_returncode=code,worker_returncodes=codes,complete=not(code or any(codes))))
    for f in handles:f.close()
    if code or any(codes):raise RuntimeError('experiment incomplete; inspect retained stage errors')


if __name__=='__main__':main()
