"""Exploratory, trajectory-clustered reanalysis of actual lane image residuals.

No controller/CP model is changed. The old 199 calibration trajectories are
explicitly repurposed as diagnostic development data; the old 100 holdout
trajectories are reused audit data, NOT a new blind confirmation set.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .tasks import load_task,RESULTS

RIDGE=1e-4
BOOTSTRAP_SEED=202611181941
BOOTSTRAP_DRAWS=5000
FEATURES=['1','e/.85','heading/.30','(e/.85)^2','(e/.85)*(heading/.30)','(heading/.30)^2']
TARGETS=['next_lateral_delta_m','next_heading_delta_rad','steering_delta_rad']


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def features(states,kappa,contexts):
    e,theta=states[...,0],states[...,1]
    base=np.stack([np.ones_like(e),e,theta,e*e,e*theta,theta*theta],-1)
    return np.concatenate([base*(np.isclose(kappa,k)[...,None]) for k in contexts],-1)


def make_data(path,task):
    from Aebs.lane_keeping.policy import load_actor
    import torch
    data=dict(np.load(path,allow_pickle=False));physical=data['states'][:,:-1]
    states=physical/[.85,.30];errors=data['errors'];contexts=data['kappas']
    actor=load_actor(task.metadata['actor']['path']).double().requires_grad_(False)
    target=np.zeros(states.shape[:2]+(3,));predicted_next=np.zeros_like(physical)
    for k in task.contexts:
        ids=np.flatnonzero(np.isclose(contexts,k));s=states[ids].reshape(-1,2);e=errors[ids].reshape(-1,2)
        actual=task.step(s,e,context=k);nominal=task.nominal_step(s,context=k)
        delta=(actual-nominal)*[.85,.30]
        with torch.no_grad():
            obs=torch.tensor(np.c_[s+e,np.full(len(s),k/.01)],dtype=torch.float64)
            nom=torch.tensor(np.c_[s,np.full(len(s),k/.01)],dtype=torch.float64)
            action_delta=((actor(obs)-actor(nom))*.45).numpy().ravel()
        target[ids]=np.c_[delta,action_delta].reshape(len(ids),80,3)
        predicted_next[ids]=(actual*[.85,.30]).reshape(len(ids),80,2)
    phi=features(states,np.broadcast_to(contexts[:,None],states.shape[:2]),task.contexts)
    consistency=np.abs(predicted_next-data['states'][:,1:]).max(axis=(0,1))
    return dict(states=states,physical=physical,errors=errors,kappa=contexts,target=target,features=phi,
                consistency=consistency,episode_seeds=data['episode_seeds'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir/'transition_state_diagnostic.json'
    if output.exists():raise FileExistsError(output)
    import torch
    torch.set_num_threads(1);start=time.monotonic()
    task=load_task('lane');folder=RESULTS/'lane_robust_synthesis_20261116_v2/final_cp'
    development_path,holdout_path=folder/'calibration.npz',folder/'holdout.npz'
    protocol=dict(kind='exploratory reanalysis, not a new blind test',
        development='existing 199 calibration episodes repurposed only for this diagnostic fit',
        audit='existing 100 holdout episodes reused; no new data collection',
        original_cp_unchanged=True,cp_claim_for_diagnostic=False,
        targets=TARGETS,basis_per_curvature=FEATURES,contexts=list(task.contexts),
        ridge=RIDGE,penalty='mean frame squared loss plus fixed lambda times squared non-intercept coefficients',
        selection='no hyperparameter or model selection; all three targets reported',
        constant_baseline='separate empirical mean per curvature, fit on development only',
        statistics='one MSE and one signed mean residual per trajectory; paired cluster bootstrap of 100 trajectories',
        bootstrap_draws=BOOTSTRAP_DRAWS,bootstrap_seed=BOOTSTRAP_SEED,
        limitation='observational state-conditional predictive evidence, not a test of all distributions or a causal safety claim',
        code_sha256=digest(__file__),development_sha256=digest(development_path),audit_sha256=digest(holdout_path),
        task_metadata=task.metadata)
    # Seal the prescribed analysis before reading numerical outcomes.
    protocol_path=args.output_dir/'transition_state_diagnostic_protocol.json'
    if protocol_path.exists():raise FileExistsError(protocol_path)
    protocol_path.write_text(json.dumps(protocol,indent=2))
    dev=make_data(development_path,task);test=make_data(holdout_path,task)
    x=dev['features'].reshape(-1,18);y=dev['target'].reshape(-1,3)
    penalty=np.eye(18)*RIDGE
    for j in [0,6,12]:penalty[j,j]=0
    weights=np.linalg.solve(x.T@x/len(x)+penalty,x.T@y/len(x))
    prediction=test['features']@weights
    constant=np.zeros_like(test['target']);means={}
    for k in task.contexts:
        mean=dev['target'][np.isclose(dev['kappa'],k)].mean(axis=(0,1));means[str(k)]=mean.tolist()
        constant[np.isclose(test['kappa'],k)]=mean
    residual_constant=test['target']-constant;residual_state=test['target']-prediction
    mse_c=(residual_constant**2).mean(axis=1);mse_s=(residual_state**2).mean(axis=1)
    bias_c=residual_constant.mean(axis=1);bias_s=residual_state.mean(axis=1)
    rng=np.random.default_rng(BOOTSTRAP_SEED);ids=rng.integers(len(mse_c),size=(BOOTSTRAP_DRAWS,len(mse_c)))
    result={}
    for j,name in enumerate(TARGETS):
        gain=mse_c[:,j]-mse_s[:,j];ci=np.quantile(gain[ids].mean(1),[.025,.975])
        bias_diff=bias_c[:,j]-bias_s[:,j];bias_ci=np.quantile(bias_diff[ids].mean(1),[.025,.975])
        result[name]=dict(constant_mse=float(mse_c[:,j].mean()),state_mse=float(mse_s[:,j].mean()),
            constant_rmse=float(np.sqrt(mse_c[:,j].mean())),state_rmse=float(np.sqrt(mse_s[:,j].mean())),
            mse_reduction=float(gain.mean()),mse_reduction_ci95=ci.tolist(),
            fractional_mse_reduction=float(gain.mean()/mse_c[:,j].mean()) if mse_c[:,j].mean()>0 else None,
            improved_trajectories=int((gain>0).sum()),worsened_trajectories=int((gain<0).sum()),
            constant_signed_mean_residual=float(bias_c[:,j].mean()),state_signed_mean_residual=float(bias_s[:,j].mean()),
            signed_residual_change_ci95=bias_ci.tolist(),
            actual_delta_min=float(test['target'][...,j].min()),actual_delta_max=float(test['target'][...,j].max()))
    rows=[]
    for i in range(len(mse_c)):
        row=dict(trajectory=i,episode_seed=int(test['episode_seeds'][i]),curvature=float(test['kappa'][i]))
        for j,key in enumerate(TARGETS):
            for suffix,value in [('constant_mse',mse_c[i,j]),('state_mse',mse_s[i,j]),('mse_reduction',mse_c[i,j]-mse_s[i,j]),
                                 ('constant_signed_residual',bias_c[i,j]),('state_signed_residual',bias_s[i,j])]:
                row[key+'_'+suffix]=float(value)
        rows.append(row)
    csv_path=args.output_dir/'transition_state_diagnostic_per_trajectory.csv'
    with csv_path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    examples=[]
    # Predeclared deterministic illustrations: first held-out episode in each
    # curvature, t=0 and t=40. No selection of the largest or best fit.
    for k in task.contexts:
        i=int(np.flatnonzero(np.isclose(test['kappa'],k))[0])
        for t in [0,40]:
            examples.append(dict(trajectory=i,time_index=t,curvature=float(k),state_physical=test['physical'][i,t].tolist(),
                semantic_error=test['errors'][i,t].tolist(),actual_delta=test['target'][i,t].tolist(),
                constant_prediction=constant[i,t].tolist(),state_prediction=prediction[i,t].tolist(),
                predicted_steering_shift_degrees=float(np.rad2deg(prediction[i,t,2]-constant[i,t,2])),
                predicted_heading_shift_degrees=float(np.rad2deg(prediction[i,t,1]-constant[i,t,1]))))
    report=dict(protocol=protocol,development_trajectories=len(dev['states']),audit_trajectories=len(test['states']),
                horizon=80,results=result,constant_means=means,weights=weights.tolist(),examples=examples,
                dynamics_reconstruction_max_error_m_rad=test['consistency'].tolist(),
                framewise_state_model_vs_constant_predicted_shift=dict(
                    heading_mean_abs_deg=float(np.rad2deg(np.abs(prediction[...,1]-constant[...,1]).mean())),
                    steering_mean_abs_deg=float(np.rad2deg(np.abs(prediction[...,2]-constant[...,2]).mean()))),
                elapsed_seconds=time.monotonic()-start,
                interpretation='State prediction improvement suggests a state-independent transition mean can lose information in this frozen visual system. It does not establish a true iid noise law, invalidate SPVC theory, imply unsafe actual trajectories, or furnish a new CP guarantee.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(dict(output=str(output),results=result)),flush=True)


if __name__=='__main__':main()
