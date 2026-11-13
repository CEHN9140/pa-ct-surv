BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
LOG_FILE="${EXPERIMENT}/logs/residual_pathology_control.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/residual_pathology_control" || -e "${EXPERIMENT}/checkpoints/residual_pathology_control" ]]; then
  echo "Matched pathology control already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup setsid /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/residual_pathology_control.py" \
  --teacher_run "${EXPERIMENT}/results/residual_teacher" \
  --num_workers 8 \
  --results_root "${EXPERIMENT}/results/residual_pathology_control" \
  --checkpoint_root "${EXPERIMENT}/checkpoints/residual_pathology_control" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started matched pathology controls, PID: $!"
echo "Log: ${LOG_FILE}"
