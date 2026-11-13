BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUNTIME="${EXPERIMENT}/.phenotype_runtime/bin/python"
LOG_FILE="${EXPERIMENT}/logs/shape_ablation.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/shape_ablation" ]]; then
  echo "Shape ablation already exists; refusing to overwrite"
  exit 1
fi
if [[ ! -x "${RUNTIME}" ]]; then
  echo "Missing isolated phenotype runtime; see README for setup"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 nohup setsid "${RUNTIME}" -u "${EXPERIMENT}/shape_ablation.py" \
  --predicted_shape_run "${EXPERIMENT}/results/predicted_shape_increment" \
  --results_root "${EXPERIMENT}/results/shape_ablation" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started Shape stage1 ablation (CPU only), PID: $!"
echo "Log: ${LOG_FILE}"
