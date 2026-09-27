# Semantic-only SPVC

This is a deliberately minimal variant of the original AEBS SPVC pipeline.

Only the observation front end changes:

```text
original: state + z -> cGAN image -> state_net -> PPO -> dynamics -> SBC
this version: real image -> SemanticEncoder -> PPO -> dynamics -> SBC
```

The original PPO architecture/checkpoint, SBC architecture, SBC losses,
transition noise, VT grid, IBP verifier, probability thresholds and alternating
training loop are unchanged. The later safety filter, recoverable-set, conformal
contract, standalone-PPO repair and scenario/QP work are not used.

During VT training and verification, the grid state is already
`[normalized semantic distance, speed]`, so it is passed directly to the PPO.
The old four-dimensional cGAN latent argument remains in the function signature
for compatibility but is ignored.

## Fast check

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.smoke_test
```

## Run the original SPVC learner/verifier with the semantic front end

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.run \
  --timeout-seconds 3600
```

Outputs are written to `results/semantic_spvc_minimal/`.

This version is intentionally a controlled baseline. It does not strengthen the
verification argument and does not claim robustness to semantic estimation error.

## Separate QP prototype (Q1 only)

The new `qp_constraint.py`, `qp_layer.py`, and `check_qp.py` are an isolated
one-step diagnostic. They do not modify the baseline `run.py` training path.
Run the unit checks and sampled diagnostic yourself in the server `vt` environment:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m pytest -q tests/test_semantic_spvc_qp.py
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.check_qp \
  --checkpoint-dir results/semantic_spvc_my_run \
  --output-dir results/semantic_spvc_qp_q1 --samples 128
```

The QP uses a linearized midpoint estimate of the original discrete SBC
expectation. The diagnostic separately checks the unlinearized midpoint
expectation and the original VT verifier's IBP upper bound. A successful QP
solve alone is not a new SBC certificate.

After Q1, diagnose only its positive-residual/slack states (no training):

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.diagnose_qp_q1 \
  --checkpoint-dir results/semantic_spvc_my_run \
  --q1-dir results/semantic_spvc_qp_q1 \
  --output-dir results/semantic_spvc_qp_q1b
```

It separates action clipping from additional QP changes, checks whether each
case lies in the original VT decrease-check region, and scans legal actions for
midpoint/IBP residuals. The finite action scan is diagnostic, not a proof of
feasibility or infeasibility over continuous actions.

The first uniform Q1 sample missed the original verifier's decrease-check
region entirely. To sample that region directly, without training:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.check_qp \
  --checkpoint-dir results/semantic_spvc_my_run \
  --scope original-verifier --samples 128 \
  --output-dir results/semantic_spvc_qp_q1c_region
```

This still checks only sampled grid centers; it does not transfer the original
certificate to the QP policy.

Q1c exposed that the original VT "IBP upper" can be smaller than the value at
a noise-cell midpoint. Before any closed-loop QP work, audit the original
interval API call against a proper single-input `BoundedTensor` interval:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.audit_ibp
```

This reads the existing 29 Q1c states and checkpoints only. Until the audit is
resolved, do not interpret the legacy `0/10000` or probability number as a
validated safety guarantee.

Q1d confirmed that the legacy value equals the noise-cell lower-corner point
value, not an interval upper bound. To recheck the saved PPO/SBC on the original
100x100 grid with correctly constructed intervals, without retraining:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.recheck_corrected
```

The fixed verifier is separate from `Aebs/VT/verify.py`; old files and numbers
are kept for comparison. This recheck alone still does not audit all theorem
assumptions or image-to-semantics error.

The first corrected full-grid recheck found one hard decrease violation in the
21 state centers selected by the original filter. To locate it and compare the
frozen nominal/QP actions (still no training or rollout):

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.diagnose_corrected_violation
```

The one corrected-IBP violation has midpoint residual about -1.972 but
10x10-cell interval residual about +1.972. Before retraining, hold the model,
action and noise support fixed and refine only the interval partition:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.refine_noise_ibp
```

The violating point's corrected IBP residual becomes negative at 40x40 noise
cells. Recheck the entire original state grid with that same partition, saving
separately from the 10x10 result:

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.recheck_corrected \
  --noise-bins 40 \
  --output-dir results/semantic_spvc_corrected_recheck_noise40
```

The user elected to continue the original SPVC experimental path for now,
without requiring that optional corrected-grid recheck. The legacy verifier
files remain untouched. For the next minimal frozen-QP paired rollout on the
original AEBS environment (empirical only):

```bash
cd /root/work_based_on_spvc/artical-F122
/opt/miniconda3/bin/conda run --no-capture-output -n vt \
  python -m Aebs.semantic_spvc.evaluate_qp_original
```

This uses true semantic state like VT training, not the real-image path, and
does not turn the legacy `0/10000` or `96.151%` into validated guarantees.

## Differentiable QP as the policy output (joint training path)

`qp_policy.py` wraps the semantic PPO so its final action is the differentiable
SBC-QP solution. `train_qp_joint.py` reuses the original VT learner's L/P
losses and optimizers; it saves compatible PPO and SBC checkpoints. A bounded
smoke and one original-grid-sized training pass were run on the server:

```bash
python -m Aebs.semantic_spvc.train_qp_joint \
  --output-dir results/semantic_spvc_qp_joint_full_20260926 \
  --train-states 0 --batch-size 64 --l-epochs 10 --p-epochs 1
python -m Aebs.semantic_spvc.evaluate_qp_original \
  --checkpoint-dir results/semantic_spvc_qp_joint_full_20260926 \
  --output-dir results/semantic_spvc_qp_joint_full_20260926_rollout
python -m Aebs.semantic_spvc.check_trained_qp \
  --checkpoint-dir results/semantic_spvc_qp_joint_full_20260926 \
  --output-dir results/semantic_spvc_qp_joint_full_20260926_image_smoke
```

The complete path trained, saved, passed a real-image forward check, and the
legacy VT check reported 1/10000. However the paired 20 episodes were all
unsafe for both PPO and PPO+QP; QP made no action change beyond clipping.
This is pipeline completion, not a successful safety result. The old VT
interval-call limitation still applies to its reported violation count.

The improved, explicitly different fine-tuning run keeps the original PPO
initialization and reduces only the actor learning rate in this new QP entry
point (the original VT code/default LR are unchanged):

```bash
python -m Aebs.semantic_spvc.train_qp_joint \
  --resume-dir results/semantic_spvc_qp_joint_full_20260926 \
  --output-dir results/semantic_spvc_qp_original_init_lr1e4_full_20260926 \
  --original-controller-init --train-states 0 --l-epochs 10 --p-epochs 1 \
  --p-learning-rate 0.0001
python -m Aebs.semantic_spvc.evaluate_qp_original \
  --checkpoint-dir results/semantic_spvc_qp_original_init_lr1e4_full_20260926 \
  --output-dir results/semantic_spvc_qp_original_init_lr1e4_full_20260926_rollout
python -m Aebs.semantic_spvc.diagnose_qp_feasibility \
  --checkpoint-dir results/semantic_spvc_qp_original_init_lr1e4_full_20260926 \
  --output-dir results/semantic_spvc_qp_original_init_lr1e4_full_20260926_feasibility
```

This produced 20/20 successful paired rollouts, a 4.16% QP action-change
rate, and a legacy 0/10000 grid count. However 35.07% of QP steps had
positive slack. Only 2/260 steps on three trajectories lay in the original
verifier's narrow SBC-value band. The legacy count and these empirical
rollouts do not establish a safety certificate, and image input has only a
forward-pass smoke test, not a closed-loop evaluation.
