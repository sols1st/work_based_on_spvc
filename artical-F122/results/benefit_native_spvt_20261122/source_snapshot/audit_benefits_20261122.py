"""Independent metric, CP-rank, dynamics and paired-scenario audit."""
from pathlib import Path
import json,math
import numpy as np
import torch
from .proxy_architecture_compare import write
from .benefit_native_spvt import dynamics,target

AEBS=Path('results/benefit_cp_aebs_20261122')
SPVT=Path('results/benefit_native_spvt_20261122')


def bootstrap_paired(difference,seed=202611229301):
    # First average across frozen training seeds, then resample scenarios.
    per_scenario=difference.mean(0)
    rng=np.random.default_rng(seed)
    means=np.concatenate([per_scenario[rng.integers(len(per_scenario),size=(100,len(per_scenario)))].mean(1) for _ in range(50)])
    return dict(mean=float(per_scenario.mean()),ci95_percentile=np.percentile(means,[2.5,97.5]).tolist(),n_scenarios=len(per_scenario),resamples=5000)


def audit():
    a=json.loads((AEBS/'summary.json').read_text())
    thresholds=json.loads((AEBS/'frozen_thresholds.json').read_text())
    for name,row in a.items():
        with np.load(AEBS/name/'calibration.npz') as cal:
            q=float(np.sort(cal['scores'])[474])
        with np.load(AEBS/name/'test.npz') as test:
            assert q==thresholds[name]['q_m']==row['q_m']
            assert int((test['scores']<=q).sum())==row['covered']
            for kind,n in row['outcomes'].items():assert int((test['outcomes']==kind).sum())==n
            assert len(test['scores'])==2000
            np.testing.assert_allclose(test['episode_mae_m'].mean(),row['trajectory_mae_m']['mean'])
    seeds=[27411,27412,27413]
    base=np.stack([np.load(AEBS/f'gray32_mlp_{s}_brightness_0p7_current_robust/test.npz')['episode_mae_m'] for s in seeds])
    cur=np.stack([np.load(AEBS/f'gray32_spatial_{s}_brightness_0p7_current_robust/test.npz')['episode_mae_m'] for s in seeds])
    aebs_mae=bootstrap_paired((base-cur)*100)
    before=np.stack([np.load(AEBS/f'gray32_mlp_{s}_blur_1p0_original_ppo/test.npz')['outcomes']=='success' for s in seeds])
    after=np.stack([np.load(AEBS/f'gray32_mlp_{s}_blur_1p0_current_robust/test.npz')['outcomes']=='success' for s in seeds])
    completion=bootstrap_paired((after.astype(float)-before.astype(float))*100)
    s=json.loads((SPVT/'summary.json').read_text())
    for key,row in s.items():
        with np.load(SPVT/key/'calibration.npz') as cal:q=float(np.sort(cal['scores'])[474])
        assert q==row['q_deg']
        with np.load(SPVT/key/'test.npz') as test:
            states=test['states_cm_degrees']
            err=test['action_error_deg']
            np.testing.assert_allclose(abs(err).max(1),test['scores'])
            assert int((test['scores']<=q).sum())==row['covered']
            unsafe=(abs(states[:,:,0])>80).any(1)
            assert int(unsafe.sum())==row['unsafe']
            rmse=np.sqrt((states[:,1:,0]**2).mean(1))
            np.testing.assert_allclose(rmse.mean(),row['rmse_cm'],rtol=1e-6)
            for tol in [5,10,20,40]:
                assert int(((abs(states[:,40:,0]).max(1)<=tol)&~unsafe).sum())==row[f'precision_{tol}cm']
            # Replay every stored one-step transition and action-error score.
            x=torch.from_numpy(states[:,:-1].reshape(-1,2))
            action=torch.from_numpy(test['actions'].reshape(-1))
            nxt=dynamics(x,action).numpy().reshape(states[:,1:].shape)
            np.testing.assert_allclose(nxt,states[:,1:],atol=3e-5,rtol=1e-6)
            error=((action-target(x))*30).numpy().reshape(err.shape)
            np.testing.assert_allclose(error,err,atol=2e-5,rtol=1e-5)
    pairs={}
    for mode in ['iid','persistent']:
        for native in ['spvt_pid','spvt_epoch100']:
            b=np.load(SPVT/f'{mode}__{native}/test.npz')['states_cm_degrees']
            b=np.sqrt((b[:,1:,0]**2).mean(1))
            cs=[np.load(SPVT/f'{mode}__current_lowdim_{seed}/test.npz')['states_cm_degrees'] for seed in [2026112211,2026112212,2026112213]]
            c=np.stack([np.sqrt((v[:,1:,0]**2).mean(1)) for v in cs])
            pairs[mode+'__'+native]=bootstrap_paired(b-c)
    write(AEBS/'independent_audit.json',dict(arms=len(a),episodes=84*2499,cp_rank=475,all_metrics_recomputed=True,
        dark_scene_mae_improvement_cm=aebs_mae,blur_completion_improvement_percentage_points=completion))
    write(SPVT/'independent_audit.json',dict(arms=len(s),episodes=16*2499,cp_rank=475,all_metrics_and_transitions_recomputed=True,
        paired_rmse_improvement_cm=pairs))


if __name__=='__main__':
    torch.set_num_threads(1)
    audit()
