# Lane keeping (2-D visual observation)

2026-11-06: original image PPO now has a constructive ReLU neural SBC. See results/lane_keeping_20261106_neural_sbc_certificate_v1/README.md. The loader restores the final Tanh. Legacy LQR reports used reversed intervals and are not valid certificate evidence; archived files remain unchanged.

A self-contained lane-keeping task for the SPVC/VT pipeline.  It does not
depend on CARLA, X-Plane or PyBullet: the road is rendered with numpy + Pillow,
and the semantic observation is a frozen DINOv2 CLS feature plus a small
projection head.

## Modules

```text
core.py             dynamics, safety set, coordinates (v=4 m/s, dt=.1, H=80)
renderer.py         perspective road renderer (192x192, optional OOD knobs)
env.py              semantic and observation-noise Gym environments
data.py             scene-group dataset generation
perception.py       frozen DINO + 2-D projection; deployment observer
train_perception.py projection training
cp.py               split-conformal 2-D box calibration
policy.py           deterministic SB3 actor export/load
residual.py         LQR + bounded neural residual; quadratic SBC
train_ppo.py        semantic / noise-aware PPO
train_residual.py   PPO for the certified residual class
certify_linear.py   adaptive interval certificate (LQR and residual class)
sbc.py              nonnegative SBC and interval helpers
train_sbc.py        SBC training with counterexample buffer
verify.py           interval audit (state grid x observation-error box)
evaluate.py         rendered-image closed-loop evaluation
```

## Quick start

Run from `artical-F122`:

```bash
../.venv-vision-sls/bin/python -m Aebs.lane_keeping.data \
  --output-dir data/lane_keeping_v1 --scenes 24 --states-per-scene 40 --image-size 192

../.venv-vision-sls/bin/python -m Aebs.lane_keeping.train_perception \
  --data-dir data/lane_keeping_v1 --output-dir results/lane_run/perception --epochs 300

../.venv-vision-sls/bin/python -m Aebs.lane_keeping.cp \
  --data-dir data/lane_keeping_v1 \
  --projection results/lane_run/perception/projection.pt \
  --output results/lane_run/cp/cp_95.json
```

See `results/lane_keeping_20261105/README.md` for the full experiment, exact
hyperparameters and the current failure analysis.  The old freely trained Tanh SBC remains uncertified. The new ReLU SBC passes two numerical continuous-box audits for the restored PPO, conditional on its observation-error box.
