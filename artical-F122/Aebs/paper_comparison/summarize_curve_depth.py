"""Replay checkpoints and measure paired error changes for the 5D proxy study."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .curve_proxy_depth import BUDGETS
from .summarize_proxy_depth import summarize
from .proxy_architecture_compare import SEEDS, write


def main(root):
    summary = summarize(root,'curve_braking',budgets=BUDGETS,metric_scale=[100,180/np.pi,100])
    write(root/'summary.json',summary)
    path=root/str(max(BUDGETS))/'curve_braking'
    test=np.load(path/'inputs.npz')['split']=='test'
    candidate=summary[max(BUDGETS)]['selected_nn']
    def errors(arch):
        keys=[arch] if arch=='fixed_Cx' else [f'{arch}_{s}' for s in SEEDS]
        return np.stack([abs(np.load(path/(k+'.npz'))['residual_physical'][test])*[100,180/np.pi,100] for k in keys])
    gain=errors('fixed_Cx')-errors(candidate)
    per_frame=gain.mean(0)
    rng=np.random.default_rng(2026112235)
    draws=np.concatenate([per_frame[rng.integers(len(per_frame),size=(100,len(per_frame)))].mean(1) for _ in range(50)])
    write(root/'paired_test.json',dict(candidate=candidate, reference='fixed_Cx', gain_mean=per_frame.mean(0).tolist(),
          ci95_percentile=np.percentile(draws,[2.5,97.5],axis=0).tolist(), gain_by_seed=gain.mean(1).tolist(),
          scope='Paired static frames, three training seeds fixed; not trajectory coverage or retraining uncertainty.',
          n_frames=int(test.sum()),resamples=5000,units=['cm','degree','cm']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    main(args.root)
