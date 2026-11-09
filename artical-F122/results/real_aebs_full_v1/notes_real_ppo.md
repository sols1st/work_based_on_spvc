# New real-image PPO: behavior only, not certified

A new PPO was trained on the real-image observation (`real_aebs_artifact_v3`): 100k steps, 14,172 train images, mixed nominal/image episodes.

- Observation: test MAE 0.0874 m; CP radius 0.2307 m; test coverage 93.33%.
- Behavior: nominal 400/400, dense real-image replay 400/400, test-only sparse library 400/400 unsafe (lookup error up to 0.878 m).
- SBC: random ReLU training collapsed to a constant (~12.75); warmup from the old SBC reached a region-passing point but the real-policy tube still had ~2,300 positive drift states, so no certificate. The transferred old actor/SBC is the certified package.
- This is retained as a negative/behavioral result, not as a certificate.
