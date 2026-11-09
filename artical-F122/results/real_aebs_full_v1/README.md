# AEBS with real 640x640 RGB images: full experiment

2026-11-08. **Status: follows the original SPVC reporting skeleton (observation model -> PPO -> nonnegative SBC -> continuous verification -> conditional safety probability). The improvement isolated here is the real-RGB front-end; the previously certified PPO/SBC pair is kept fixed so the front-end effect is not confounded with controller retraining.**

## Scope (read first)

- The **visual front-end is real 640x640 RGB**: frozen DINOv2-S/14 CLS 384 -> 384-128-128-1 distance projection. No 32x32 grayscale image is used.
- The **controller and SBC are the previously certified AEBS pair** (`vision_sls_spvc_sbc_pass_20261029`), kept fixed to isolate the front-end change. The actor is the certified PPO; the SBC is its nonnegative `[2,16,8,1]` network, scaled by 1.1 so the unsafe lower bound crosses 20 (old fine-grid lower 19.04).
- This is therefore a **front-end replacement certificate**, not a certificate for a newly trained real-image PPO. A real-image PPO was also trained and reaches 400/400 nominal/dense behavior, but its SBC training did not close; see `notes_real_ppo.md` below.
- The tube proof is conditional on every image observation lying in the CP ball. CP test coverage is 93.33%, not 95%; 6.7% of held-out images are outside the tube. The replay proof is for a frozen 400-image library.

## Data

- `/root/work_based_on_spvc/carla_data_spvc_expanded/full01`: 24,000 PNG, 640x640 RGB, 4 weather x 3 target colors, labels `distance_m` 5-16 m.
- Official scene-group split: train 14,172, validation 4,908, test 4,920; split by 0.25 m distance bins.
- Calibration = all validation bins; test = official held-out test bins; train/calibration/test disjoint.

## Pipeline

1. **Observation**: frozen DINOv2-S/14 CLS 384 -> 384-128-128-1 MLP -> normalized distance. Test MAE **0.0874 m**, max 0.525 m; calibration MAE 0.0807 m.
2. **CP envelope**: constant Bernstein envelope `b0=1`; split-CP 95% q=0.072493 normalized, radius **0.2307 m**. Calibration coverage 95.03%; test coverage 93.33%.
3. **Actor**: transferred old certified AEBS PPO, inputs `[d_hat/3.1827, measured speed]`; same dynamics/reward/initial/unsafe domains.
4. **SBC**: transferred old nonnegative `[2,16,8,1]` ReLU SBC, final layer scaled 1.1; initial upper **0**, unsafe lower **27.4625**.
5. **Tube verification**: all live `d in (6,16]`, `v in (1e-6,3]`, observation errors inside the CP ball; two physical state dimensions, no curvature; adaptive interval to depth 15.
6. **Replay verification**: 400 evenly spaced frozen real images, Voronoi nearest-image lookup, all continuous initial states.
7. **Behavior**: 20x20 initial grid (400 starts), max 400 steps, existing terminal semantics.

## Results

| Check | Result |
|---|---|
| Tube continuous check | 303,574 cells checked; 150,153 proved; 3,682 threshold-excluded; **0 actual witnesses; 0 unresolved**; depth 15 |
| Replay 400-image check | 159,350 cells; 87,047 proved; **0 witnesses; 0 unresolved**; depth 12 |
| Regions | initial upper **0.0**; unsafe lower **27.4625** (≥20) |
| Behavior nominal | **400/400 success** |
| Behavior dense real images | **400/400 success**; lookup max 0.003 m |
| Behavior test-only images | **400/400 success**; lookup max 0.878 m |
| CP calibration/test | 95.03% / **93.33%** |

The tube result is the same guarantee as the original AEBS SBC: nonnegative B, `B(initial)<=1`, `B(unsafe)>=20`, and no increase for any observation-tube action; it supports the stopped-process risk bound `B0/20 <= 5%`. The replay result is deterministic for the frozen library and does not need the CP support assumption.

## Limitations

- **Transferred actor/SBC**: the certificate is for the old certified AEBS controller; it is not a newly trained real-image PPO certificate. The real-image PPO artifact (`real_aebs_artifact_v3`) is behavior evidence only.
- **CP test coverage is 93.33%**, below the nominal 95%; the tube proof is conditional on the tube. 6.7% of test images are outside and are not covered by the tube argument.
- **SBC scaling**: old SBC unsafe lower was 19.04; scaling by 1.1 gives 27.46. Scaling is monotone and the full tube/replay verification was rerun; this is disclosed, not hidden.
- **Replay is a 400-image library**, not live CARLA and not a new-image guarantee.
- **Numerical**: float64 interval arithmetic; no outward-rounded IEEE proof; no real-vehicle claim.
- **Behavior** 400/400 is evidence, not a reachability proof.

## Files

- `policy.pt`, `observation.pt`, `sbc.pt`: real-image observation, transferred certified actor, scaled SBC.
- `perception_contract.json`: CP coefficients/q and coverage.
- `observations.npz`: 24,000 real-image distances/observations for replay/behavior.
- `verification/tube/report.json`: tube proof.
- `verification/replay_400.json`: 400-image replay proof.
- `behavior_real.json`: nominal/dense/test behavior.
- `deployment_check.json`: raw PNG -> DINO -> real observation -> actor; 6 images, finite actions.
- `metrics.json`: all numbers and hashes.
- `notes_real_ppo.md`: what happened with the new real-image PPO (not certified).
- `source_snapshot/`: real_artifact, test encoder, verification and SBC helpers.

## Reproduce

Server, `vt` environment:

```bash
cd /root/work_based_on_spvc/artical-F122
python -m Aebs.vision_sls.verify_sbc_cells \
  --artifact-dir results/real_aebs_full_v1 \
  --barrier-dir results/real_aebs_full_v1 \
  --perception-contract results/real_aebs_full_v1/perception_contract.json \
  --observation-model tube --max-depth 28 --max-cells 5000000 \
  --output-dir results/real_aebs_full_v1/verification/tube_repro
```

Replay uses the 400-image `observations.npz` from `real_aebs_full_replay_v1` and `--observation-model replay`.
