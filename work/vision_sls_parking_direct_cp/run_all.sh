#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$TASK_ROOT/.venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=8
export MPLBACKEND=Agg LIBGL_ALWAYS_SOFTWARE=1
SOURCE="$TASK_ROOT/source"
WEIGHTS="$TASK_ROOT/assets/dinov2_vitg14_pretrain.pth"
DINO="$TASK_ROOT/assets/dinov2"
mkdir -p "$TASK_ROOT/results"
visual() {
  xvfb-run -a -s '-screen 0 1280x1024x24 +extension GLX +render -noreset' \
    "$PYTHON" -u "$TASK_ROOT/run_visual_rollout.py" --source "$SOURCE" \
    --weights "$WEIGHTS" --dinov2-source "$DINO" "$@"
}
if [[ ! -f "$TASK_ROOT/results/calibration/diagnostic_traces.npz" ]]; then
  visual --output "$TASK_ROOT/results/calibration" --rollouts 40 \
    --seed 2026102503 --random-only > "$TASK_ROOT/results/calibration.log" 2>&1
fi
"$PYTHON" "$TASK_ROOT/check_model.py"
"$PYTHON" -u "$TASK_ROOT/calibrate_direct_cp.py" \
  --traces "$TASK_ROOT/results/calibration/diagnostic_traces.npz" \
  --output "$TASK_ROOT/results" --protocol "$TASK_ROOT/protocol.json" \
  > "$TASK_ROOT/results/direct_cp_calibration.log" 2>&1
"$PYTHON" -u "$TASK_ROOT/plan_direct_cp.py" --source "$SOURCE" \
  --calibration "$TASK_ROOT/results/direct_cp_calibration.json" \
  --output "$TASK_ROOT/results/direct_cp_plan" > "$TASK_ROOT/results/direct_cp_plan.log" 2>&1
CP_RADIUS="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["radius"])' "$TASK_ROOT/results/direct_cp_calibration.json")"
visual --output "$TASK_ROOT/results/direct_cp" --rollouts 50 --seed 42 \
  --plan "$TASK_ROOT/results/direct_cp_plan/data/unicycle_of_perception_0.npz" \
  --direct-cp-radius "$CP_RADIUS" > "$TASK_ROOT/results/direct_cp.log" 2>&1
"$PYTHON" "$TASK_ROOT/compare_results.py" > "$TASK_ROOT/results/comparison.log" 2>&1
"$PYTHON" -m pip freeze > "$TASK_ROOT/results/environment.freeze.txt"
"$PYTHON" "$TASK_ROOT/record_provenance.py"
"$PYTHON" "$TASK_ROOT/write_report.py"
echo 'Complete: results/direct_cp_comparison.json'
