# AEBS trajectory-level conformal calibration

This package implements the first, deliberately minimal CP experiment for a
frozen semantic-SPVC controller.  One complete trajectory is one calibration
sample.  Time steps from the same trajectory are never treated as IID.

The common registered scope is:

- initial distance `Uniform(15,16)` metres;
- initial speed `Uniform(2.5,3.0)` m/s;
- horizon `H=400`;
- deterministic original AEBS dynamics;
- frozen PPO and frozen PPO+SBC-QP from the same checkpoint.

The observation model is selected explicitly and creates a new experiment:

- `exact`: exact simulator distance and speed;
- `dataset_nearest`: at every step, use the real dataset image whose label is
  nearest to the true distance, run the semantic encoder, and feed its point
  estimate plus exact speed to the controller;
- `uniform_contract`: draw an independent distance error uniformly inside the
  saved state-conditional interval at every step. This is a deliberately harsh
  synthetic stress distribution, not the empirical camera-error distribution.

The trajectory score is positive exactly when the trajectory enters the
configured unsafe set (`distance <= 6 m` and `speed > 0.5 m/s`).  A
nonpositive conformal threshold therefore supports a finite-horizon,
distribution-specific probabilistic statement.  It is not an infinite-horizon
or distribution-free-over-states safety certificate.

Run the unit checks:

```bash
python -m pytest -q tests/test_conformal_controller.py
```

Run the registered 99% safety / 99% confidence experiment:

```bash
python -m Aebs.conformal.calibrate_controller \
  --calibration-episodes 459 --test-episodes 200 \
  --epsilon 0.01 --beta 0.01 \
  --output-dir results/conformal/trajectory_cp_99_99_final_20260927
```

Run the real-image nearest-frame replay as a separate registered distribution:

```bash
python -m Aebs.conformal.calibrate_controller \
  --semantic-mode dataset_nearest \
  --calibration-episodes 459 --test-episodes 200 \
  --epsilon 0.01 --beta 0.01 \
  --output-dir results/conformal/trajectory_cp_image_replay_99_99_20260927
```

Run the synthetic interval stress test:

```bash
python -m Aebs.conformal.calibrate_controller \
  --semantic-mode uniform_contract \
  --calibration-episodes 459 --test-episodes 200 \
  --epsilon 0.01 --beta 0.01 \
  --output-dir results/conformal/trajectory_cp_uniform_semantic_99_99_20260927
```

The script refuses to overwrite a nonempty output directory.  It writes the
configuration before rollout, hashes the semantic encoder and frozen
checkpoints (and the image dataset when used), and stores all
per-trajectory scores and outcomes for audit.
