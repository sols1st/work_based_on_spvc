"""One fixed projection adaptation experiment on fresh OpenGL images.

Frozen DINO-G/14, fixed original C, unchanged camera/alpha/dynamics. Spatial
cell holdout, train-cell-only historical replay, validation-only selection;
test descriptors/evaluation are deferred until the selected checkpoint exists.
No PPO/SBC training, CP fitting, arbitrary rescaling, or automatic deployment.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from .core import RENDER_OFFSET
from .perception import load_projection, fitted_bound, residual_summary
from .test_perception import sha


DOMAIN_LOW = np.array([-3., -1.5])
DOMAIN_HIGH = np.array([0., 1.5])


def cell_ids(states):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 4 or not np.isfinite(states).all():
        raise ValueError('finite [N,4] states required')
    xy = states[:, :2]
    valid = ((xy >= DOMAIN_LOW) & (xy <= DOMAIN_HIGH)).all(1)
    ij = np.floor((xy-DOMAIN_LOW)/(DOMAIN_HIGH-DOMAIN_LOW)*8).astype(int).clip(0, 7)
    result = ij[:, 0]*8 + ij[:, 1]
    return np.where(valid, result, -1)


def make_split_plan(per_cell=16, seed=7):
    if not isinstance(per_cell, int) or per_cell < 1:
        raise ValueError('positive integer per_cell required')
    rng = np.random.default_rng(seed)
    order = rng.permutation(64)
    assignment = np.empty(64, dtype='<U10')
    assignment[order[:40]] = 'train'
    assignment[order[40:52]] = 'validation'
    assignment[order[52:]] = 'test'
    states, cells, splits = [], [], []
    width = (DOMAIN_HIGH-DOMAIN_LOW)/8
    for cell in range(64):
        ij = np.array([cell//8, cell % 8])
        low = DOMAIN_LOW+ij*width
        xy = low + rng.uniform(1e-5, 1-1e-5, (per_cell, 2))*width
        s = np.column_stack([xy, rng.uniform(-.75*np.pi, .75*np.pi, per_cell), np.zeros(per_cell)])
        states.append(s); cells.extend([cell]*per_cell); splits.extend([assignment[cell]]*per_cell)
    states = np.vstack(states)
    cells, splits = np.asarray(cells), np.asarray(splits)
    if not np.array_equal(cell_ids(states), cells):
        raise AssertionError('plan/cell coordinates disagree')
    return states, cells, splits


def historical_replay_indices(source_states, cells, splits, maximum=512, seed=7):
    """Never train on historical features from fresh validation/test cells."""
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError('positive maximum required')
    candidates = np.flatnonzero(np.isin(cell_ids(source_states), np.unique(cells[splits == 'train'])))
    if not len(candidates):
        raise ValueError('no historical train-cell samples')
    return np.random.default_rng(seed).choice(candidates, min(maximum, len(candidates)), replace=False)


def choose_epoch(scores):
    """Scores are validation MSE only. Test metrics must never be passed here."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('finite nonempty validation score sequence required')
    return int(np.argmin(scores))


def train_candidate(initial, train_features, train_targets, validation_features, validation_targets,
                    replay_features, replay_targets, epochs=120, learning_rate=1e-4, seed=7):
    import torch
    model = copy.deepcopy(initial)
    model.requires_grad_(False)
    model.net.requires_grad_(True)
    c_before = model.C.detach().clone()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=learning_rate)
    n = len(train_features)
    rng = np.random.default_rng(seed)
    scores, history = [], []
    with torch.no_grad():
        scores.append(float(((model(validation_features)-validation_targets)**2).mean()))
    best_score, best = scores[0], copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs+1):
        model.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, 128):
            ids = order[start:start+128]
            replay_ids = rng.choice(len(replay_features), min(128, len(replay_features)), replace=False)
            loss_new = ((model(train_features[ids])-train_targets[ids])**2).mean()
            loss_replay = ((model(replay_features[replay_ids])-replay_targets[replay_ids])**2).mean()
            loss = loss_new + .25*loss_replay
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            score = float(((model(validation_features)-validation_targets)**2).mean())
        if not np.isfinite(score): raise ValueError('nonfinite validation score')
        scores.append(score)
        if score < best_score:
            best_score, best = score, copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            row = dict(epoch=epoch, mean_loss=float(np.mean(losses)), validation_mse=score)
            history.append(row)
            print('projection epoch=%d loss=%.7f validation_RMSE=%.7f' %
                  (epoch, row['mean_loss'], np.sqrt(score)), flush=True)
    model.load_state_dict(best)
    if not torch.equal(c_before, model.C): raise AssertionError('C changed during adaptation')
    model.eval().requires_grad_(False)
    return model, dict(best_epoch=choose_epoch(scores), validation_mse_by_epoch=scores,
                       history=history, C_bitwise_unchanged=True, test_used_for_selection=False,
                       learning_rate=learning_rate, epochs=epochs, replay_weight=.25)


def main():
    import torch
    import h5py
    from PIL import Image, ImageDraw
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    from .renderer import ParkingRenderer

    p = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[3]
    p.add_argument('--vision-sls-root', type=Path, default=project/'VISION-SLS')
    p.add_argument('--dino-repo', type=Path, default=project/'external/dinov2')
    p.add_argument('--dino-weights', type=Path, default=project/'external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--per-cell', type=int, default=16)
    p.add_argument('--epochs', type=int, default=120)
    args = p.parse_args()
    data = args.vision_sls_root/'expe/parking_lot/data'
    checkpoint = data/'unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt'
    source_file = data/'unicycle_10000_512small_03width_background4_075w_3pi4_op097_dino_1536_states_large_val.pt'
    bound_file = data/'unicycle_observability_bound_ppoly_097_300000_val_3_0.97_0.04.npz'
    required = [checkpoint, source_file, bound_file, args.dino_weights, args.dino_repo/'hubconf.py']
    if args.output_dir.exists() or args.per_cell < 1 or args.epochs < 1 or any(not f.is_file() for f in required):
        p.error('new output directory, positive counts, complete frozen inputs required')
    if not torch.cuda.is_available(): p.error('this registered experiment requires CUDA')
    torch.set_num_threads(1); torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    states, cells, splits = make_split_plan(args.per_cell)
    sources = {str(f): sha(f) for f in required}
    sources.update({str(f): sha(f) for f in Path(__file__).parent.glob('*.py')})
    scene_files = [f for f in (args.vision_sls_root/'expe/parking_lot/plane_v2').glob('*') if f.is_file()]
    sources.update({str(f): sha(f) for f in scene_files})
    args.output_dir.mkdir(parents=True)
    config = dict(source_sha256=sources, seed=7, renderer='GUI/OpenGL via Xvfb/Mesa',
                  states=len(states), cells=64, xy_grid=[8, 8],
                  split_cells=dict(train=40, validation=12, test=12),
                  split_samples={s: int((splits == s).sum()) for s in np.unique(splits)},
                  world_domain=dict(x=[-3, 0], y=[-1.5, 1.5], heading=[-.75*np.pi, .75*np.pi], speed=0),
                  per_cell=args.per_cell, epochs=args.epochs, learning_rate=1e-4, batch_size=128,
                  objective='new RGB feature MSE to fixed Cx + .25 historical train-cell MSE',
                  replay_max_samples=512, DINO_frozen=True, C_frozen=True,
                  test_descriptors_before_checkpoint=False, PPO_training=False, SBC_training=False,
                  selection='validation MSE only; includes original epoch 0 candidate',
                  test_role='one evaluation after checkpoint selection; no test-driven retraining')
    (args.output_dir/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    np.savez_compressed(args.output_dir/'split_plan.npz', states=states, cell_ids=cells, splits=splits)
    started = time.monotonic()
    result = dict(status='failed', training='small projection only', PPO_tested=False, SBC_tested=False,
                  formal_guarantee=False, independent_CP_calibration=False, old_AEBS_changed=False)
    renderer, cache = None, None
    try:
        initial, c, _ = load_projection(checkpoint, 'cuda')
        baseline_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
        src_states, src_features = torch.load(source_file, map_location='cpu')
        src_states = np.column_stack([src_states.numpy(), np.zeros(len(src_states))])-RENDER_OFFSET
        replay_ids = historical_replay_indices(src_states, cells, splits)
        np.save(args.output_dir/'historical_replay_indices.npy', replay_ids)
        cache = h5py.File(args.output_dir/'paired_data.h5', 'w')
        images = cache.create_dataset('rgb', (len(states), 512, 512, 3), dtype='u1',
                                      chunks=(1, 512, 512, 3), compression='gzip', compression_opts=1)
        cache.create_dataset('states', data=states)
        cache.create_dataset('cell_ids', data=cells)
        cache.create_dataset('splits', data=splits.astype('S10'))
        feats = cache.create_dataset('dino_features', (len(states), 1536), dtype='f4', fillvalue=np.nan)
        encoded = cache.create_dataset('encoded', data=np.zeros(len(states), dtype='u1'))
        visible = np.zeros(len(states), dtype=int)
        hashes, duplicate_cross_split = {}, []
        renderer = ParkingRenderer(args.vision_sls_root, 'opengl')
        for i, state in enumerate(states):
            rgb = renderer.render(state)
            images[i] = rgb; visible[i] = renderer.last_visible_robot_pixels
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            if digest in hashes and splits[hashes[digest]] != splits[i]:
                duplicate_cross_split.append([hashes[digest], i])
            hashes.setdefault(digest, i)
            if (i+1) % 128 == 0: print('captured OpenGL %d/%d' % (i+1, len(states)), flush=True)
        renderer.close(); renderer = None
        cache.create_dataset('visible_robot_pixels', data=visible)
        cache.flush()
        if duplicate_cross_split:
            raise ValueError('identical RGB across spatial splits: %s' % duplicate_cross_split[:5])
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        print('loading frozen DINO-G/14', flush=True)
        backbone = torch.hub.load(str(args.dino_repo), 'dinov2_vitg14', source='local', pretrained=False)
        backbone.load_state_dict(torch.load(args.dino_weights, map_location='cpu'), strict=True)
        backbone.to('cuda').eval().requires_grad_(False)

        def encode(rows):
            with torch.inference_mode():
                for start in range(0, len(rows), 4):
                    ids = rows[start:start+4]
                    inputs = torch.stack([preprocess(Image.fromarray(images[int(i)])) for i in ids]).cuda()
                    f = backbone.forward_features(inputs)['x_norm_clstoken'].cpu().numpy()
                    if f.shape[1] != 1536 or not np.isfinite(f).all(): raise ValueError('invalid DINO features')
                    for row, value in zip(ids, f): feats[int(row)] = value; encoded[int(row)] = 1
                    if (start+4) % 128 == 0:
                        print('encoded current phase %d/%d' % (min(start+4, len(rows)), len(rows)), flush=True)
            cache.flush()

        encode(np.flatnonzero(splits != 'test'))
        if np.asarray(encoded[:])[splits == 'test'].any(): raise AssertionError('test encoded too early')
        targets = ((states+RENDER_OFFSET)@c.T).astype('float32')

        def tensor(a): return torch.as_tensor(a, dtype=torch.float32, device='cuda')
        train_rows, val_rows = np.flatnonzero(splits == 'train'), np.flatnonzero(splits == 'validation')
        replay_x = src_features[replay_ids].float().cuda()
        replay_y = tensor((src_states[replay_ids]+RENDER_OFFSET)@c.T)
        adapted, info = train_candidate(initial, tensor(feats[train_rows]), tensor(targets[train_rows]),
                                         tensor(feats[val_rows]), tensor(targets[val_rows]),
                                         replay_x, replay_y, epochs=args.epochs)
        path = args.output_dir/'projection_adapted.pt'
        torch.save({k: v.detach().cpu() for k, v in adapted.state_dict().items()}, path)
        selected_sha = sha(path)
        selection = dict(checkpoint_sha256=selected_sha, best_epoch=info['best_epoch'],
                         test_features_encoded=int(np.asarray(encoded[:])[splits == 'test'].sum()),
                         selection_complete_before_test=True)
        if selection['test_features_encoded'] != 0: raise AssertionError('test leakage')
        (args.output_dir/'selection_before_test.json').write_text(json.dumps(selection, indent=2))
        cache.attrs['selected_checkpoint_sha256_before_test_encoding'] = selected_sha
        cache.flush()
        print('checkpoint selected at epoch %d; encoding untouched test now' % info['best_epoch'], flush=True)
        encode(np.flatnonzero(splits == 'test'))
        if sha(path) != selected_sha: raise AssertionError('checkpoint changed during test')
        for k, v in initial.state_dict().items():
            if not torch.equal(v.cpu(), baseline_state[k]): raise AssertionError('baseline model changed')
        bounds = np.load(bound_file, allow_pickle=False)
        radii = fitted_bound(states, bounds['theta_poly'], float(bounds['theta_bias'][0]))
        summaries, predictions = {}, {}
        for split in ['train', 'validation', 'test']:
            rows = np.flatnonzero(splits == split)
            x = tensor(feats[rows])
            with torch.inference_mode():
                base_y, new_y = initial(x).cpu().numpy(), adapted(x).cpu().numpy()
            summaries[split] = dict(baseline=residual_summary(base_y, states[rows], c, radii[rows]),
                                     adapted=residual_summary(new_y, states[rows], c, radii[rows]))
            predictions[split] = dict(rows=rows, baseline=base_y, adapted=new_y)
            for name, mask in [('no_car_segmentation', visible[rows] == 0), ('car_in_segmentation', visible[rows] > 0)]:
                if mask.any():
                    summaries[split][name] = dict(
                        baseline=residual_summary(base_y[mask], states[rows][mask], c, radii[rows][mask]),
                        adapted=residual_summary(new_y[mask], states[rows][mask], c, radii[rows][mask]))
        base, new = summaries['test']['baseline'], summaries['test']['adapted']
        mean_better = new['latent_linf_mean'] < base['latent_linf_mean']
        p95_better = new['latent_linf_p95'] < base['latent_linf_p95']
        max_not_worse = new['latent_linf_max'] <= base['latent_linf_max']+1e-6
        decision = ('development_mean_and_tail_improved' if mean_better and p95_better and max_not_worse
                    else 'mean_improved_but_tail_or_p95_not_improved' if mean_better else 'no_mean_improvement')
        result.update(status=decision, test=summaries['test'], splits=summaries, training=info,
                      checkpoint=str(path), checkpoint_sha256=selected_sha,
                      test_evaluated_after_selection=True, test_used_for_model_selection=False,
                      historical_replay_count=len(replay_ids), cross_split_duplicate_rgb=0,
                      DINO_frozen=True, original_projection_unchanged=True,
                      split_cells_disjoint=True, C_unchanged=True, old_bound_is_reference_only=True,
                      not_claimed='PPO task performance, CP coverage or SBC certificate',
                      visibility_definition='segmentation only; not a proof of no RGB signal')
        np.savez_compressed(args.output_dir/'test_predictions.npz',
                            states=states[splits == 'test'], targets=targets[splits == 'test'],
                            baseline=predictions['test']['baseline'], adapted=predictions['test']['adapted'],
                            visible_robot_pixels=visible[splits == 'test'])
        sheet = Image.new('RGB', (8*160, 8*185), 'white'); draw = ImageDraw.Draw(sheet)
        for cell in range(64):
            i = cell*args.per_cell
            x, y = (cell % 8)*160, (cell//8)*185
            sheet.paste(Image.fromarray(images[i]).resize((160, 160)), (x, y+25))
            draw.text((x+2, y+2), 'cell %02d %s' % (cell, splits[i]), fill='black')
        sheet.save(args.output_dir/'spatial_split_contact_sheet.png')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
        for ax, metric, factor, title in [
            (axes[0], 'latent_linf_mean', 1., 'Mean observation error (unitless)'),
            (axes[1], 'pose_xy_euclidean_mae_m', 100., 'Position error via inverse C (cm)')]:
            values = [factor*base[metric], factor*new[metric]]
            bars = ax.bar(['Original projection', 'Adapted projection'], values, color=['#b86b60', '#427a68'])
            ax.bar_label(bars, fmt='%.3f', padding=3); ax.set_title(title); ax.set_ylim(0, max(values)*1.2)
        fig.suptitle('%d fresh spatial-holdout images: no PPO/SBC claim' % int((splits == 'test').sum()))
        fig.savefig(args.output_dir/'test_comparison.png', dpi=160); plt.close(fig)
    except BaseException as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        if renderer is not None: renderer.close()
        if cache is not None: cache.close()
        result['runtime_seconds'] = time.monotonic()-started
        (args.output_dir/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('[Parking projection-only adaptation; no PPO/SBC]')
    print('status=%s best_epoch=%d' % (result['status'], info['best_epoch']))
    print('test mean observation error %.6f -> %.6f; p95 %.6f -> %.6f; max %.6f -> %.6f' %
          (base['latent_linf_mean'], new['latent_linf_mean'], base['latent_linf_p95'], new['latent_linf_p95'],
           base['latent_linf_max'], new['latent_linf_max']))
    print('test position MAE %.3f -> %.3f cm' % (base['pose_xy_euclidean_mae_m']*100, new['pose_xy_euclidean_mae_m']*100))
    print('old empirical envelope coverage %.2f%% -> %.2f%% (reference, NOT new CP)' %
          (100*base['empirical_bound_coverage'], 100*new['empirical_bound_coverage']))
    print('metrics:', args.output_dir/'metrics.json')


if __name__ == '__main__': main()
