#!/usr/bin/env bash
# Reproduce one parking-lot initial condition using the author's supplied models.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$TASK_ROOT/.venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=8 MPLBACKEND=Agg LIBGL_ALWAYS_SOFTWARE=1
export PYTHONPATH="$TASK_ROOT/source"
mkdir -p "$TASK_ROOT/results"
cd "$TASK_ROOT/source/expe/parking_lot"
echo 'Planning the nominal trajectory and output-feedback controller...'
"$PYTHON" -u main_parking_lot.py > "$TASK_ROOT/results/plan.log" 2>&1
echo 'Checking the saved plan...'
"$PYTHON" -u "$TASK_ROOT/check_plan.py" --source "$TASK_ROOT/source" \
  --output "$TASK_ROOT/results" > "$TASK_ROOT/results/plan_checks.log" 2>&1
echo 'Fitting a separate diagnostic error envelope...'
"$PYTHON" -u "$TASK_ROOT/refit_error_envelope.py" --source "$TASK_ROOT/source" \
  --output "$TASK_ROOT/results/calibration" > "$TASK_ROOT/results/calibration_run.log" 2>&1
echo 'Running 50 image-based closed-loop simulations...'
xvfb-run -a -s '-screen 0 1280x1024x24 +extension GLX +render -noreset' \
  "$PYTHON" -u "$TASK_ROOT/run_visual_rollout.py" --source "$TASK_ROOT/source" \
  --weights "$TASK_ROOT/assets/dinov2_vitg14_pretrain.pth" \
  --dinov2-source "$TASK_ROOT/assets/dinov2" \
  --output "$TASK_ROOT/results/visual_rollouts" --rollouts 50 \
  > "$TASK_ROOT/results/visual_rollouts.log" 2>&1
"$PYTHON" -m pip freeze > "$TASK_ROOT/results/environment.freeze.txt"
"$PYTHON" "$TASK_ROOT/record_provenance.py"
"$PYTHON" "$TASK_ROOT/build_report.py"
echo 'Experiment complete. See results/visual_rollouts/summary.json.'
