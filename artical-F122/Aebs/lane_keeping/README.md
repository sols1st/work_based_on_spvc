# Lane keeping (2-D visual observation)

2026-11-07: trajectory_cp.py now calibrates complete 80-step image rollouts with frozen models. 199 calibration runs give q=0.0900931580; 95/100 independent holdout runs are fully covered (old 0.08 covers 82/100). The original learned SBC passes affine and IBP at the new radius. See [trajectory CP package](../../results/lane_trajectory_cp_20261107_v1/README.md). Coverage is marginal under the registered nominal rollout law, not a fixed-calibration conditional or real-world guarantee.

Current result (later 2026-11-06): AEBS NonnegativeSBC [2,16,8,1] trained from random initialization, no geometric teacher. Two SBC seeds pass; published seed passes affine and IBP audits. Original PPO already verifies, so actor feedback is implemented/tested but was not triggered. See [learned certificate package](../../results/lane_keeping_20261106_aebs_method_sbc_v1/README.md).

New entry points: synthesize.py (learner/verifier/feedback loop), train_joint.py (AEBS-class SBC), refine_actor.py (SBC-driven actor update and task rollback), verify_learned.py (continuous-box audit). Previous constructive result below remains a baseline.

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
hyperparameters and historical failure analysis. The old Tanh checkpoints remain uncertified; the new AEBS-class Tanh/ReLU model from train_joint.py passes both numerical continuous-box audits. The earlier constructive ReLU certificate remains a separate baseline. All guarantees are conditional on the declared model and observation-error box.
