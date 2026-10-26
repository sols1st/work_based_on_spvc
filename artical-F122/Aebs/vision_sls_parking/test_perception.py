"""Frozen projection/scene accuracy diagnostic, not PPO/SBC/CP certification.

Paired targets always use the SAME captured physical state. No nearest-image
lookup, no training, no deleting hidden images, no model or noise-law change.
"""
import argparse
import json
import time
from pathlib import Path
import hashlib
import numpy as np
from .core import C, RENDER_OFFSET, INITIAL_STATES
from .perception import (load_projection, fitted_bound, residual_summary,
                         identifiability_witness)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_panel(count=128, seed=7):
    rng = np.random.default_rng(seed)
    random_states = rng.uniform([-3., -1.5, -.75*np.pi, 0],
                                [0., 1.5, .75*np.pi, 0], (count, 4))
    hidden_probes = np.array([[-1.7, -.3, 0, 0], [-1.3, -.3, 0, 0],
                              [-1.7, .3, 0, 0], [-1.3, .3, 0, 0]])
    return (np.vstack([INITIAL_STATES, random_states, hidden_probes]),
            np.array(['original_start']*15+['uniform_paper_domain']*count+['occlusion_probe']*4))


def main():
    import torch
    from PIL import Image, ImageDraw
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    p = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[3]
    p.add_argument('--vision-sls-root', type=Path, default=project/'VISION-SLS')
    p.add_argument('--dino-repo', type=Path, default=project/'external/dinov2')
    p.add_argument('--dino-weights', type=Path,
                   default=project/'external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--features-only', action='store_true')
    p.add_argument('--render-only', action='store_true', help='cache exact paired RGB and occlusion witnesses, no DINO inference')
    p.add_argument('--panel-cache', type=Path, help='reuse this exact fixed-panel RGB cache, verified against registered states')
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--panel-source', choices=['registered', 'supplied_features'], default='registered')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--renderer', choices=['tiny', 'opengl', 'egl'], default='tiny')
    args = p.parse_args()
    data = args.vision_sls_root/'expe/parking_lot/data'
    ck = data/'unicycle_checkpoint_observability_ft_075w_3pi4_097_300000_post.pt'
    saved_data = data/'unicycle_10000_512small_03width_background4_075w_3pi4_op097_dino_1536_states_large_val.pt'
    bound_path = data/'unicycle_observability_bound_ppoly_097_300000_val_3_0.97_0.04.npz'
    required = [ck, saved_data, bound_path]
    if args.features_only and args.render_only:
        p.error('features-only and render-only are mutually exclusive')
    if args.panel_cache is not None:
        required += [args.panel_cache]
    if not (args.features_only or args.render_only):
        required += [args.dino_weights, args.dino_repo/'hubconf.py']
    if args.output_dir.exists() or args.samples < 1 or args.batch_size < 1 or any(not f.is_file() for f in required):
        p.error('new output dir, positive counts and all matching upstream inputs required')
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable; select --device cpu explicitly')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    provenance = {str(f): sha(f) for f in required}
    provenance.update({str(f): sha(f) for f in Path(__file__).parent.glob('*.py')})
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'config.json').write_text(json.dumps(dict(
        source_sha256=provenance, features_only=args.features_only, render_only=args.render_only,
        random_samples=args.samples, panel_source=args.panel_source, seed=7, renderer=args.renderer,
        preprocessing='PIL RGB -> bicubic Resize(224) -> ToTensor -> ImageNet Normalize; no center crop',
        model='official DINOv2-G/14 1536-D CLS, float32, frozen',
        panel=('without-replacement supplied-state subset; source indices saved in samples.npz'
               if args.panel_source == 'supplied_features'
               else '15 starts + %d uniform paper-domain states + 4 occlusion probes' % args.samples),
        pose_target='C_normalized @ (same captured world state + [3,0,0,0])',
        torch_version=torch.__version__, training=False), indent=2), encoding='utf-8')
    started = time.monotonic()
    result = dict(status='failed', training=False, PPO_tested=False, SBC_tested=False,
                  formal_contract=False, fresh_holdout=False, AEBS_comparison=False)
    renderer = None
    try:
        projection, c, info = load_projection(ck, args.device)
        bound = np.load(bound_path, allow_pickle=False)
        if int(bound['degree']) != 3 or int(bound['nx']) != 2:
            raise ValueError('unsupported bound polynomial; do not guess monomial ordering')
        coef, bias = bound['theta_poly'], float(bound['theta_bias'][0])
        sx, features = torch.load(saved_data, map_location='cpu')
        if sx.ndim != 2 or sx.shape[1] != 3 or features.shape != (len(sx), 1536):
            raise ValueError('unexpected saved feature/state layout')
        source_states = np.column_stack([sx.numpy(), np.zeros(len(sx))])-RENDER_OFFSET
        with torch.inference_mode():
            saved_y = projection(features.to(args.device).float()).cpu().numpy()
        result['checkpoint'] = info
        result['supplied_features'] = residual_summary(
            saved_y, source_states, c, fitted_bound(source_states, coef, bias))
        print('[Supplied upstream features; NOT a fresh holdout]', result['supplied_features'], flush=True)
        np.savez_compressed(args.output_dir/'supplied_feature_residuals.npz',
                            states=source_states, predicted=saved_y,
                            target=(source_states+RENDER_OFFSET)@c.T)
        if args.features_only:
            result.update(status='supplied_features_diagnostic_complete', rgb_inference_done=False)
        else:
            from .renderer import ParkingRenderer
            source_indices = None
            if args.panel_source == 'supplied_features':
                if args.samples > len(source_states):
                    raise ValueError('not enough supplied states for without-replacement panel')
                source_indices = np.random.default_rng(7).choice(len(source_states), args.samples, replace=False)
                states = source_states[source_indices]
                labels = np.array(['supplied_feature_matched']*len(states))
            else:
                states, labels = make_panel(args.samples)
            if args.panel_cache is not None:
                cache = np.load(args.panel_cache, allow_pickle=False)
                if (not np.array_equal(cache['states'], states) or not np.array_equal(cache['groups'], labels)
                        or str(cache['renderer']) != args.renderer
                        or cache['rgb'].shape != (len(states), 512, 512, 3)
                        or cache['rgb'].dtype != np.uint8 or cache['visible'].shape != (len(states),)):
                    raise ValueError('cached RGB/state/settings differ from registered panel')
                rgb, visible, pose_gap = cache['rgb'], cache['visible'], cache['pose_gap']
            else:
                renderer = ParkingRenderer(args.vision_sls_root, args.renderer)
                rgb, visible, pose_gap = [], [], []
                for i, state in enumerate(states):
                    rgb.append(renderer.render(state))
                    visible.append(renderer.last_visible_robot_pixels)
                    pose_gap.append(renderer.last_pose_error)
                    if (i+1) % 32 == 0:
                        print('rendered %d/%d' % (i+1, len(states)), flush=True)
                rgb, visible = np.asarray(rgb), np.asarray(visible)
                renderer.close(); renderer = None
                np.savez_compressed(args.output_dir/'rendered_panel.npz',
                                    rgb=rgb, states=states, groups=labels, visible=visible,
                                    pose_gap=np.asarray(pose_gap), renderer=args.renderer)
            targets = (states+RENDER_OFFSET) @ c.T
            witnesses = identifiability_witness(states, rgb, targets)
            # Crucial evidence exists independently of any neural inference.
            (args.output_dir/'identifiability.json').write_text(json.dumps(dict(
                witnesses=witnesses, scope='finite rendered panel, deterministic single-image encoder'), indent=2))
            if args.render_only:
                result.update(status='visibility_diagnostic_complete', rgb_inference_done=False,
                              renderer=args.renderer, input_count=len(states),
                              no_car_segmentation_count=int((visible == 0).sum()),
                              visibility_definition='segmentation mask only; translucent geometry may still reveal RGB signal',
                              identical_rgb_different_target_witnesses=witnesses[:5],
                              renderer_pose_max_error_m=float(max(pose_gap)))
                print('[Visibility diagnostic]', result, flush=True)
                print('Paired RGB cache saved; frozen DINO inference not run.', flush=True)
                return
            preprocess = transforms.Compose([
                transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
            print('loading frozen official DINOv2-G/14 (1536D)', flush=True)
            backbone = torch.hub.load(str(args.dino_repo), 'dinov2_vitg14', source='local', pretrained=False)
            backbone.load_state_dict(torch.load(args.dino_weights, map_location='cpu'), strict=True)
            backbone.to(args.device).eval().requires_grad_(False)
            predictions, descriptors = [], []
            with torch.inference_mode():
                for begin in range(0, len(rgb), args.batch_size):
                    batch = torch.stack([preprocess(Image.fromarray(im))
                                         for im in rgb[begin:begin+args.batch_size]]).to(args.device)
                    feature = backbone.forward_features(batch)['x_norm_clstoken']
                    if feature.shape[1] != 1536:
                        raise ValueError('wrong DINO dimension; never substitute 384-D backbone')
                    predictions.append(projection(feature).cpu().numpy())
                    descriptors.append(feature.cpu().numpy())
                    print('inferred %d/%d' % (min(begin+args.batch_size, len(rgb)), len(rgb)), flush=True)
            y = np.vstack(predictions)
            new_features = np.vstack(descriptors)
            b = fitted_bound(states, coef, bias)
            result['new_rgb'] = residual_summary(y, states, c, b)
            if source_indices is not None:
                old_features = features[source_indices].numpy()
                cosine = (new_features*old_features).sum(1)/(
                    np.linalg.norm(new_features, axis=1)*np.linalg.norm(old_features, axis=1))
                gap = np.abs(y-saved_y[source_indices]).max(1)
                result['same_state_source_comparison'] = dict(
                    source_indices=source_indices.tolist(),
                    supplied_feature_metrics=residual_summary(saved_y[source_indices], states, c, b),
                    rendered_rgb_metrics=result['new_rgb'],
                    feature_cosine_mean=float(cosine.mean()), feature_cosine_p05=float(np.quantile(cosine, .05)),
                    projection_output_gap_linf_mean=float(gap.mean()),
                    projection_output_gap_linf_p95=float(np.quantile(gap, .95)),
                    projection_output_gap_linf_max=float(gap.max()),
                    note='identical pose/target, supplied features vs rerendered images; not a new independent test')
            groups = dict(no_car_segmentation=visible == 0, car_in_segmentation=visible > 0)
            groups.update({str(label): labels == label for label in np.unique(labels)})
            result['groups'] = {name: residual_summary(y[mask], states[mask], c, b[mask])
                                for name, mask in groups.items() if mask.any()}
            result['no_car_segmentation_count'] = int((visible == 0).sum())
            result['visibility_definition'] = 'segmentation mask only; translucent geometry may still reveal RGB signal'
            result['identical_rgb_different_target_witnesses'] = witnesses[:5]
            np.savez_compressed(args.output_dir/'samples.npz', states=states, groups=labels,
                                visible_robot_pixels=visible, predicted=y, target=targets,
                                residual=y-targets, features=new_features, bounds=b,
                                source_indices=(source_indices if source_indices is not None else np.array([], dtype=int)))
            error = np.abs(y-targets).max(axis=1)
            ids = list(range(min(15, len(states))))+np.argsort(error)[-5:][::-1].tolist()
            sheet = Image.new('RGB', (5*256, 4*290), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, i in enumerate(ids):
                x, yy = (j % 5)*256, (j//5)*290
                sheet.paste(Image.fromarray(rgb[i]).resize((256, 256)), (x, yy+34))
                draw.text((x+3, yy+1), '#%d error %.3f bound %.3f' % (i, error[i], b[i]), fill='black')
                draw.text((x+3, yy+16), 'visible car pixels %d' % visible[i], fill='black')
            sheet.save(args.output_dir/'perception_contact_sheet.png')
            for name, r in result['groups'].items():
                print('[%s] %s' % (name, r), flush=True)
            result.update(status='frozen_perception_diagnostic_complete', rgb_inference_done=True,
                          renderer=args.renderer, renderer_pose_max_error_m=float(max(pose_gap)),
                          input_count=len(states), dino_frozen=True, projection_frozen=True,
                          pixel_equivalent_to_original_opengl=False if args.renderer == 'tiny' else None,
                          interpretation='empirical observation error, not task performance or safety',
                          next_step='inspect visibility/render shift before choosing PPO observation/history')
        # Large immutable DINO weights were hashed at start; check only metadata
        # now; no training op, optimizer, or source-file writes exist in this run.
    except BaseException as exc:
        result['error'] = '%s: %s' % (type(exc).__name__, exc)
        raise
    finally:
        if renderer is not None:
            renderer.close()
        result['runtime_seconds'] = time.monotonic()-started
        (args.output_dir/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('[VISION-SLS parking frozen perception diagnostic]')
    print('status=%s; no training/PPO/SBC/CP guarantee' % result['status'])
    if 'new_rgb' in result:
        r = result['new_rgb']
        print('RGB n=%d latent L_inf mean=%.6f p95=%.6f max=%.6f empirical bound coverage=%.2f%%' %
              (r['samples'], r['latent_linf_mean'], r['latent_linf_p95'], r['latent_linf_max'],
               100*r['empirical_bound_coverage']))
        print('pose position MAE=%.6f m; heading MAE=%.6f rad; no_car_segmentation=%d' %
              (r['pose_xy_euclidean_mae_m'], r['pose_heading_wrapped_mae_rad'], result['no_car_segmentation_count']))
    print('metrics: %s' % (args.output_dir/'metrics.json'))


if __name__ == '__main__': main()
