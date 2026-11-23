#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/../../.."
experiment=tests/fold_difficulty_analysis/conch_sample_learning_curve
mkdir -p "$experiment/results/private/logs"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin="$PWD/.radiomics_env/bin/python"
"$python_bin" -u "$experiment/abmil_curve.py" > "$experiment/results/private/logs/abmil.log" 2>&1 &
abmil_pid=$!
"$python_bin" -u "$experiment/ridge_curve.py" > "$experiment/results/private/logs/titan.log" 2>&1 &
ridge_pid=$!
printf 'ABMIL PID=%s; Ridge PID=%s\n' "$abmil_pid" "$ridge_pid"
wait "$abmil_pid"; abmil_exit=$?
wait "$ridge_pid"; ridge_exit=$?
if [ "$abmil_exit" -ne 0 ] || [ "$ridge_exit" -ne 0 ]; then
  printf 'FAILED: ABMIL exit=%s, Ridge exit=%s\n' "$abmil_exit" "$ridge_exit"
  exit 1
fi
"$python_bin" "$experiment/report.py"
