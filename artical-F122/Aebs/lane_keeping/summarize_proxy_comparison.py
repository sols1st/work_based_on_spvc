"""Summarize all preregistered pairs, without selecting a favorable seed."""
import argparse,json
from pathlib import Path
import numpy as np
from .learned_sbc import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    protocol=json.loads((root/'protocol.json').read_text());rows=[];pairing=[]
    for seed in protocol['training_seeds']:
        static=json.loads((root/f'train_{seed}/report.json').read_text())
        assert static['linear']['minibatch_sequence_sha256']==static['neural']['minibatch_sequence_sha256']
        holds={}
        for kind in ['linear','neural']:
            cpdir=root/f'cp_{seed}_{kind}';r=json.loads((cpdir/'report.json').read_text());s=static[kind];test=s['metrics']['test']
            held=dict(np.load(cpdir/'holdout.npz',allow_pickle=False));cal=dict(np.load(cpdir/'calibration.npz',allow_pickle=False));holds[kind]=held
            assert len(held['scores'])==100 and len(cal['scores'])==199
            assert sha256(cpdir/'actor.pt')==protocol['actor_sha256'] and sha256(cpdir/'sbc.pt')==protocol['sbc_sha256']
            assert sha256(cpdir/'observation.pt')==s['checkpoint_sha256']
            cert=json.loads((cpdir/'verification/certificate.json').read_text())
            rows.append(dict(seed=seed,kind=kind,best_epoch=s['best_epoch'],test_lateral_mae_m=test['decoded_mae_physical'][0],
                test_heading_mae_rad=test['decoded_mae_physical'][1],test_decoded_mse=test['decoded_mse'],
                test_p95_normalized_error=test['decoded_score_quantiles'][1],test_raw_latent_l2_mean=test['raw_latent_l2_mean'],
                holdout_score_mean=float(held['scores'].mean()),holdout_score_p95=float(np.quantile(held['scores'],.95)),holdout_score_max=float(held['scores'].max()),
                cp_radius=r['contract']['radius'],covered_holdout=r['holdout']['covered'],
                unsafe_calibration=r['unsafe_calibration'],unsafe_holdout=r['unsafe_holdout'],certificate_status=r['certificate_status'],
                actual_witnesses=sum(x['actual_witnesses'] for x in cert['checks']),unresolved=sum(x['unresolved'] for x in cert['checks']),
                initial_B_upper=cert['initial']['bound'],unsafe_B_lower=cert['unsafe']['bound'],
                holdout_mean_abs_lateral_m=float(abs(held['states'][:,:-1,0]).mean()),
                holdout_mean_action_change=float(abs(np.diff(held['actions'],axis=1)).mean()),
                holdout_by_curvature={str(k):dict(n=int((held['kappas']==k).sum()),covered=int((held['scores'][held['kappas']==k]<=r['contract']['radius']).sum()),unsafe=int(held['unsafe'][held['kappas']==k].sum())) for k in np.unique(held['kappas'])},
                shared_verified_design_radius=.1,cp_to_design_radius_slack=.1-r['contract']['radius'],
                geometry=s['geometry'],optimization_seconds=s['optimization_seconds'],optimizer_updates=s['optimizer_updates']))
        assert np.array_equal(holds['linear']['states'][:,0],holds['neural']['states'][:,0])
        assert np.array_equal(holds['linear']['kappas'],holds['neural']['kappas'])
        pair=rows[-2:];d={k:pair[1][k]-pair[0][k] for k in ['test_lateral_mae_m','test_heading_mae_rad','test_decoded_mse','cp_radius','holdout_mean_abs_lateral_m','holdout_mean_action_change']}
        pairing.append(dict(seed=seed,neural_minus_linear=d,cp_radius_reduction_percent=100*(pair[0]['cp_radius']-pair[1]['cp_radius'])/pair[0]['cp_radius']))
    metrics=['test_lateral_mae_m','test_heading_mae_rad','test_decoded_mse','cp_radius','holdout_score_p95','covered_holdout','holdout_mean_abs_lateral_m','holdout_mean_action_change','optimization_seconds']
    summary={}
    for kind in ['linear','neural']:
        selected=[r for r in rows if r['kind']==kind]
        summary[kind]={k:dict(mean=float(np.mean([r[k] for r in selected])),sample_sd=float(np.std([r[k] for r in selected],ddof=1))) for k in metrics}
        summary[kind]['certificate_passes']=sum(r['certificate_status']=='passed_numerical' for r in selected)
        summary[kind]['unsafe_episodes_total']=sum(r['unsafe_calibration']+r['unsafe_holdout'] for r in selected)
    result=dict(protocol_sha256=sha256(root/'protocol.json'),rows=rows,paired=pairing,summary=summary,
                caveat='Mean/sample SD over 3 paired optimization seeds; same images, warm start and rollout draws. Do not pool repeated draws as independent samples or infer significance from 3 seeds.')
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
