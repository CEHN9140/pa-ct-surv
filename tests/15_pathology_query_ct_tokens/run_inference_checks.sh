BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
LOG_FILE="${EXPERIMENT}/logs/inference_checks.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/inference_checks" ]]; then
  echo "Inference checks already exist; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/inference_checks.py" \
  --output_dir "${EXPERIMENT}/inference_checks" \
  --repeats 100 \
  --batch_size 16 \
  --num_workers 4 \
  --device cuda \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started inference checks, PID: $!"
echo "Log: ${LOG_FILE}"
