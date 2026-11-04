# Parking development candidate

The actor uses the original 16D observation/history input. The paired SBC is an unverified training candidate; no safety probability is certified.

Actual images: 12/15 success, 0 collision, 0 out-of-bounds episodes. These are reused development tasks, not independent testing.

From artical-F122 on the GPU server, run (choose a new evaluation output directory):

```bash
CUDA_VISIBLE_DEVICES=0 xvfb-run -a /opt/miniconda3/envs/vt/bin/python -m Aebs.vision_sls_parking.evaluate_images --ppo-dirs results/vision_sls_parking_20261102_candidate --projection results/vision_sls_parking_20261102_candidate/projection.pt --output-dir results/parking_candidate_recheck
```
