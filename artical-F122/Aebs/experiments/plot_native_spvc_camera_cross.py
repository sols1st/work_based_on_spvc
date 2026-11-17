"""Recreate the observed H5 counterexample illustration from the frozen inputs."""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from Aebs.experiments.native_spvc_camera_cross import H5, Models, tensor


@torch.inference_mode()
def plot(source):
    torch.set_num_threads(2)
    models = Models()
    with np.load(source/'frame_estimates.npz') as z:
        test = z['h5_test_indices']
        d = z['h5_distances_m'][test]
        old = z['h5_native_m'][test]
        dino = z['h5_dino_m'][test]
    j = np.abs(old - d).argmax()
    row = test[j]
    with h5py.File(H5) as h:
        camera = h['X_train'][row]
    synthetic = models.generator(torch.zeros(1,4),tensor(d[j:j+1]/models.original_scale)[:,None])
    nominal_pred = models.estimate_native(synthetic)[0] * models.original_scale
    fig = plt.figure(figsize=(9.4,3.5), layout='constrained')
    grid = fig.add_gridspec(1,3,width_ratios=(1,1,1.7))
    for i, (im,title,caption) in enumerate([
        ((synthetic[0,0].numpy()+1)/2, 'Generated image', f'Native estimate  {nominal_pred:.2f} m'),
        (camera, 'Saved CARLA image', f'Native estimate  {old[j]:.2f} m\nDINO estimate  {dino[j]:.2f} m'),
    ]):
        ax = fig.add_subplot(grid[0,i])
        ax.imshow(im, cmap='gray',interpolation='nearest',vmin=0,vmax=1)
        ax.set(xticks=[],yticks=[],title=title,xlabel=caption)
    ax = fig.add_subplot(grid[0,2])
    ax.scatter(d, old, s=13, alpha=.65, color='#c0562a', label='Original estimator')
    ax.scatter(d, dino, s=13, alpha=.7, color='#007c83', marker='x', label='DINO projection')
    ax.plot([5,16],[5,16],color='#59636c',ls='--',lw=1)
    ax.set(xlabel='Label distance (m)',ylabel='Image estimate (m)',title='All 80 saved test images')
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'H5 image {row}  |  Label {d[j]:.2f} m',fontsize=13,fontweight='bold')
    fig.savefig(source/'h5_camera_example.png',dpi=165)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    plot(p.parse_args().source)
