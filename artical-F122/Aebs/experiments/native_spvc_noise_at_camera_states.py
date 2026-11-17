"""Diagnostic: compare camera and generated observations at the same labelled distance.

This is a finite-sample probe of the released generator, separate from the
paper's additive next-state disturbance law and the registered trajectory CP.
"""
import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch

from Aebs.experiments.native_spvc_camera_cross import (
    GEN, GRAY, H5, RGB_GRAY, STATE, Models, meters, tensor, write,
)
from Aebs.vision_sls.run import sha256


def collision_counts(images, split):
    fingerprints = [hashlib.sha256(np.ascontiguousarray(row).tobytes()).digest() for row in images]
    fitted = {h for h, s in zip(fingerprints, split) if s in ('train', 'validation')}
    tested = [h for h, s in zip(fingerprints, split) if s == 'test']
    return dict(test_rows=len(tested), distinct_test_pixels=len(set(tested)),
                test_rows_matching_train_or_val_pixels=sum(h in fitted for h in tested))


def summarize(e):
    return dict(mean=float(e.mean()), p95=float(np.quantile(e,.95)), maximum=float(e.max()))


@torch.inference_mode()
def run(source, out):
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    models = Models()
    seed = 202611170514
    n = 64
    write(out/'protocol.json',dict(inputs={str(f):sha256(f) for f in [source/'protocol.json',
        source/'frame_estimates.npz',GEN,STATE,H5,RGB_GRAY,GRAY/'splits.npz',Path(__file__)]},
        rng=seed, draws=n, law='independent U[-1,1]^4 draws per frame at the exact image label',
        statistic='count observed frozen state_net(camera) within min/max of 64 generated state_net outputs; diagnostic only'))
    rng = np.random.default_rng(seed)
    with np.load(source/'frame_estimates.npz') as z:
        frames = {k:z[k] for k in z}
    results = {}
    for bank in ('h5', 'rgb'):
        ids = frames['h5_test_indices'] if bank == 'h5' else np.arange(len(frames['rgb_filenames']))
        d = frames[bank+'_distances_m'][ids].astype(np.float64)
        camera = frames[bank+'_native_m'][ids]
        nominal = models.generated_m(d)
        lo, hi = np.empty_like(d), np.empty_like(d)
        for begin in range(0,len(d),8):
            labels = d[begin:begin+8]
            latent = tensor(rng.uniform(-1.,1.,(len(labels)*n,4)))
            y = tensor(np.repeat(labels,n)/models.original_scale)[:,None]
            img = models.generator(latent,y)
            sampled = meters(models.estimate_native(img),models.original_scale).reshape(len(labels),n)
            lo[begin:begin+len(labels)] = sampled.min(axis=1)
            hi[begin:begin+len(labels)] = sampled.max(axis=1)
        inside = (lo <= camera) & (camera <= hi)
        results[bank] = dict(n=len(d), inside_sampled_range=int(inside.sum()),
            abs_camera_minus_nominal_m=summarize(np.abs(camera-nominal)),
            sampled_generated_range_width_m=summarize(hi-lo))
        np.savez_compressed(out/(bank+'.npz'), image_indices=ids, distance_m=d,
            real_estimate_m=camera, nominal_generated_m=nominal, sampled_min_m=lo, sampled_max_m=hi)
    with h5py.File(H5) as h, np.load(GRAY/'splits.npz') as s:
        split = np.full(len(h['X_train']),'',dtype='U10')
        for k in s:
            split[s[k]] = k
        results['h5']['pixel_duplicates'] = collision_counts(h['X_train'][:],split)
    with np.load(RGB_GRAY) as z:
        results['rgb']['gray32_pixel_duplicates'] = collision_counts(z['images'],z['split'])
    write(out/'report.json',results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args = parser.parse_args()
    run(args.source,args.source/'noise_diagnostic')
