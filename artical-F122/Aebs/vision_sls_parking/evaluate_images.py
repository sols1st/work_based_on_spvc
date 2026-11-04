"""Paired real-rendered-image parking evaluation with one frozen DINO load."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import C, RENDER_OFFSET
from .ppo_env import FrozenImageSensor
from .train_ppo import evaluate, sha
from .train_migration import load_ppo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppo-dirs', nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--projection', default='results/vision_sls_parking_20261025_adaptation_v1/projection_adapted.pt')
    p.add_argument('--vision-root', default='../VISION-SLS')
    p.add_argument('--dino-repo', default='../external/dinov2')
    p.add_argument('--dino-weights', default='../external/dinov2_weights/dinov2_vitg14_pretrain.pth')
    a = p.parse_args()
    out = Path(a.output_dir)
    if out.exists() or len(set(Path(s).name for s in a.ppo_dirs)) != len(a.ppo_dirs):
        p.error('new output directory and distinct model directory names required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out.mkdir(parents=True)
    cfg = dict(vars(a), projection_sha256=sha(a.projection), dino_sha256=sha(a.dino_weights),
               renderer='OpenGL', image_size=512, backbone_input=224,
               evaluation_role='15 original development scenarios, not an independent test',
               state_dimension=4, visual_dimension=3, actor_input_dimension=16,
               safety_certificate=False)
    (out/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    class Recorder(FrozenImageSensor):
        def __call__(self, state):
            y = super().__call__(state)
            records_x.append(state.copy())
            records_y.append(y.copy())
            return y
    sensor = Recorder(a.vision_root, a.dino_repo, a.dino_weights, a.projection)
    results = {}
    started = time.monotonic()
    try:
        for source in a.ppo_dirs:
            records_x, records_y = [], []
            path = Path(source)/'ppo.zip'
            checkpoint_sha = sha(path)
            model = load_ppo(path, device='cpu')
            nominal, nominal_x, nominal_u = evaluate(model, quiet=True)
            actual, actual_x, actual_u = evaluate(model, sensor)
            capture_x, capture_y = np.array(records_x), np.array(records_y)
            target = (capture_x+RENDER_OFFSET)@C.T
            error = capture_y-target
            name = Path(source).name
            np.savez_compressed(out/(name+'.npz'), proxy_states=nominal_x, proxy_actions=nominal_u,
                                image_states=actual_x, image_actions=actual_u,
                                capture_states=capture_x, observations=capture_y,
                                targets=target, residuals=error,
                                trajectory=np.repeat(np.arange(15), 31))
            results[name] = dict(checkpoint_sha256=checkpoint_sha, proxy=nominal, image=actual,
                                 latent_rmse=float(np.sqrt(np.mean(error**2))),
                                 latent_linf_p95=float(np.quantile(np.max(np.abs(error), axis=1), .95)),
                                 checkpoint_unchanged=sha(path) == checkpoint_sha)
            (out/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
            print(name, 'image success', actual['success_count'], '/15', flush=True)
    finally:
        sensor.close()
    summary = dict(config=cfg, results=results, runtime_seconds=time.monotonic()-started,
                   projection_unchanged=sha(a.projection) == cfg['projection_sha256'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
