"""Train p_theta and row-normalized C together on frozen DINO patch tokens.

Matches the active VISION-SLS train_observability.py observation loss. It
does not implement the commented-out Gramian regularizer or SLS controller.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import core
from .joint_observation import JointPatchProjection, require_invertible
from .learned_sbc import sha256


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--warm-start', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20261109)
    p.add_argument('--max-condition', type=float, default=10.)
    p.add_argument('--fixed-C', action='store_true', help='registered matched-budget ablation only')
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.max_condition < 1:
        raise ValueError('invalid training parameters')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    snapshot=args.output_dir/'source_snapshot';snapshot.mkdir()
    for src in [Path(__file__),Path(__file__).with_name('joint_observation.py')]:
        shutil.copyfile(src,snapshot/src.name)
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    saved = torch.load(args.warm_start, map_location='cpu', weights_only=False)
    if saved.get('format') != 'lane_keeping_dino_patch_projection_v1':
        raise ValueError('unexpected warm-start model')
    data = np.load(args.dataset, allow_pickle=False)
    x = torch.tensor(data['states'] / [core.E_SCALE, core.THETA_SCALE], dtype=torch.float32)
    splits = data['split']
    features = torch.load(args.features, map_location='cpu', weights_only=False)['patches'].float()
    if len(features) != len(x) or not torch.isfinite(features).all():
        raise ValueError('invalid feature alignment or values')
    tr = torch.from_numpy(np.flatnonzero(splits == 'train'))
    va = torch.from_numpy(np.flatnonzero(splits == 'validation'))
    if not len(tr) or not len(va): raise ValueError('separate train and validation required')
    model = JointPatchProjection(saved['feature_dim'], saved['hidden'], saved['grid'])
    model.visual.load_state_dict(saved['projection'])
    # Warm-start in a nonidentity coordinate system, THEN optimize both branches.
    # This rotation alone is not counted as evidence of learning C.
    C0 = torch.tensor([[3**.5 / 2, .5], [-.5, 3**.5 / 2]], dtype=torch.float32)
    with torch.no_grad():
        model.raw_C.copy_(C0)
        model.visual.out.weight.copy_(C0 @ model.visual.out.weight.clone())
        model.visual.out.bias.copy_(C0 @ model.visual.out.bias.clone())
    initial = copy.deepcopy(model.state_dict())
    if args.fixed_C:model.raw_C.requires_grad_(False)
    protocol = dict(format='lane_joint_training_protocol_v1',
        config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        bindings={k:sha256(v) for k,v in dict(dataset=args.dataset,features=args.features,
                    warm_start=args.warm_start,source=Path(__file__),
                    model_source=Path(__file__).with_name('joint_observation.py')).items()},
        feature_alignment='existing positional cache; same full dataset order; row count checked, no per-image cache digest available',
        train_rows=tr.tolist(),validation_rows=va.tolist(),
        loss='mean L2 norm of p_theta(DINO(image)) - normalized_rows(C_raw) x',
        state_scale=[core.E_SCALE,core.THETA_SCALE],initial_C=C0.tolist(),
        selection='minimum decoded normalized-state validation MSE among condition<=max_condition checkpoints',
        calibration_and_test_used_for_selection=False,
        mode='fixed-C matched-budget ablation' if args.fixed_C else 'joint-C training',
        scope='warm-start observation fitting; fixed DINO; no SLS synthesis or Gramian regularizer')
    write(args.output_dir/'protocol.json',protocol)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    best=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total=0.;grad=0.
        for ids in tr[torch.randperm(len(tr))].split(args.batch_size):
            loss=(model(features[ids])-model.nominal(x[ids])).norm(dim=1).mean()
            optimizer.zero_grad();loss.backward()
            grad+=float(model.raw_C.grad.norm()) if model.raw_C.grad is not None else 0.
            optimizer.step();total+=float(loss.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            C=model.matrix();sv=torch.linalg.svdvals(C);cond=float(sv[0]/sv[-1])
            pred=torch.cat([model(features[i]) for i in va.split(args.batch_size)])
            decoded=model.decode(pred)
            mse=float((decoded-x[va]).square().mean())
            row=dict(epoch=epoch,train_loss=total/len(tr),validation_decoded_mse=mse,
                     C=C.tolist(),condition=cond,C_gradient_norm_sum=grad)
            if np.isfinite(mse) and cond<=args.max_condition and (best is None or mse<best[0]):
                best=(mse,copy.deepcopy(model.state_dict()),epoch)
        history.append(row);write(args.output_dir/'history.json',history)
        print(json.dumps(row),flush=True)
    if best is None: raise RuntimeError('no nondegenerate finite checkpoint; do not calibrate')
    model.load_state_dict(best[1]);model.eval().requires_grad_(False)
    C=model.matrix().detach();sv=require_invertible(C,args.max_condition)
    weights=Path(saved['dino_weights'])
    checkpoint=dict(format='lane_joint_observation_v1',feature_dim=saved['feature_dim'],
        hidden=saved['hidden'],grid=saved['grid'],state_dict=model.state_dict(),C=C,
        state_scale=[core.E_SCALE,core.THETA_SCALE],max_condition=args.max_condition,
        dino_repo=saved['dino_repo'],dino_weights=str(weights),dino_model=saved['dino_model'],
        dino_weights_sha256=sha256(weights),best_epoch=best[2],
        training_protocol_sha256=sha256(args.output_dir/'protocol.json'))
    torch.save(checkpoint,args.output_dir/'joint_observation.pt')
    with torch.no_grad():
        latent=torch.cat([model(f) for f in features.split(args.batch_size)])
        decoded=model.decode(latent);err=decoded-x
    metrics={}
    for name in np.unique(splits):
        ids=np.flatnonzero(splits==name);e=err[ids]
        metrics[name]=dict(n=len(ids),decoded_mae_physical=(e.abs().mean(0)*torch.tensor([core.E_SCALE,core.THETA_SCALE])).tolist(),
                          max_normalized_error=float(e.abs().max()))
    report=dict(format='lane_joint_training_report_v1',best_epoch=best[2],
        mode=protocol['mode'],
        C_initial=C0.tolist(),C_final=C.tolist(),C_change_frobenius=float((C-C0).norm()),
        C_gradient_nonzero=any(h['C_gradient_norm_sum']>0 for h in history),
        visual_parameter_change_l2=float(sum((model.state_dict()[k]-v).square().sum()
            for k,v in initial.items() if k.startswith('visual.')).sqrt()),
        singular_values=sv.tolist(),condition=float(sv[0]/sv[-1]),metrics=metrics,
        safety_status='not certified; new trajectory CP and SBC audit required',
        joint_checkpoint_sha256=sha256(args.output_dir/'joint_observation.pt'))
    np.savez_compressed(args.output_dir/'observations.npz',latent=latent.numpy(),
                       decoded=decoded.numpy(),normalized_state=x.numpy(),split=splits,C=C.numpy())
    write(args.output_dir/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__': main()
