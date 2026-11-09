# AEBS real-RGB distance probe (development)

2026-11-08. **Scope: real 640x640 RGB images + distance labels, 1-D visual distance observation only. No PPO/SBC/closed-loop certificate.**

- Data: `/root/work_based_on_spvc/carla_data_spvc_expanded/full01` (24,000 PNG, 12 weather/color combinations, `labels.csv`).
- Features: frozen DINOv2-S/14 CLS 384 from the existing real-RGB cache `results/dino_expanded_v1/02_dino_h5/features.h5`; no 32x32 grayscale.
- Split: official 0.25 m distance bins; 26 train bins, 5 calibration bins, 4 test bins; 8,000/1,500/1,500 sampled images. Train-bin fit/early split; no test leakage.
- Head: 384 -> 128 -> 64 -> 1, GELU, standardized features; target `distance_m / 3.18272066`.
- Metrics: test MAE **0.1786 m**, RMSE 0.2370 m, P95 0.5150 m, max 0.8683 m.
- Split CP: q=0.17806 normalized, radius **0.5667 m**, calibration 95.07%, test **96.53%** coverage.
- Same probe on 32x32 grayscale (local comparison): test MAE 0.2158 m, CP radius 0.6073 m, test coverage 97.5%.
- Raw-image end-to-end check on 6 selected PNGs: 5/6 within 0.52 m; one low-sun 10 m image predicted 8.79 m (error -1.21 m).

Files: `metrics.json`, `config.json`, `predictions.csv`, `head.pt`, `raw_image_check.json`, `grayscale_same_probe.json`, `source.py`.

This is a development probe. It does not re-encode the official 9 test bins, does not run PPO/SBC, and does not establish a certificate. The real RGB CP radius is larger than the current AEBS state-dependent mean radius (~0.43 m); a full certificate must use the wider envelope or a tighter state-dependent one.
