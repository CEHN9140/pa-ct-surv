BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUNTIME="${EXPERIMENT}/.phenotype_runtime/bin/python"
LOG_FILE="${EXPERIMENT}/logs/predicted_shape_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/predicted_shape_increment" ]]; then
  echo "Predicted-Shape diagnostic already exists; refusing to overwrite"
  exit 1
fi
if [[ ! -x "${RUNTIME}" ]]; then
  echo "Missing isolated phenotype runtime; see README for setup"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 nohup setsid "${RUNTIME}" -u "${EXPERIMENT}/predicted_shape_increment.py" \
  --phenotype_run "${EXPERIMENT}/results/ct_phenotype_increment" \
  --feature_run "${EXPERIMENT}/results/feature_increment" \
  --results_root "${EXPERIMENT}/results/predicted_shape_increment" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started public MedicalNet predicted-Shape diagnostic (CPU only), PID: $!"
echo "Log: ${LOG_FILE}"
