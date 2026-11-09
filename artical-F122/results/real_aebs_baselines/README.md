# AEBS front-end baselines: cGAN / 32x32 gray / real RGB

2026-11-08. Same 4,920 real-image test set and distance labels.

| Front-end | test MAE | CP95 radius | test coverage |
|---|---:|---:|---:|
| original SPVC cGAN+state_net (old 400-image checkpoint) | 2.747 m | 5.221 m | 87.64% |
| 32x32 grayscale DINO (same 14,172 train images) | 0.1017 m | 0.2656 m | 90.47% |
| real RGB 640x640 DINO (same 14,172 train images) | **0.0874 m** | **0.2307 m** | **93.33%** |

Same actor/SBC tube under each envelope: cGAN fails (538 actual witnesses, 11,103 unresolved at depth 18); grayscale passes (0/0 at depth 15); real RGB passes (0/0 at depth 15).

The cGAN row is the original SPVC front-end checkpoint applied to the new real test set; it is **not** a same-budget cGAN retrain. The grayscale-vs-real comparison is the controlled representation comparison. The real-vs-gray improvements are 16.4% MAE, 15.1% CP radius, +2.87 pp coverage. Per-weather rain is an exception (gray is lower).

Files: `cgan_state_net.json`, `gray_dino.json`, `comparison.json`, `comparison.png`, source scripts.
