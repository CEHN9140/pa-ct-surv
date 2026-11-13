BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUNTIME="${EXPERIMENT}/.phenotype_runtime/bin/python"
LOG_FILE="${EXPERIMENT}/logs/ct_phenotype_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/ct_phenotype_increment" ]]; then
  echo "CT phenotype experiment1 already exists; refusing to overwrite"
  exit 1
fi
if [[ ! -x "${RUNTIME}" ]]; then
  echo "Missing isolated phenotype runtime; see README for setup"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 nohup setsid "${RUNTIME}" -u "${EXPERIMENT}/ct_phenotype_increment.py" \
  --source_run "${EXPERIMENT}/results/nested_risk_increment" \
  --feature_run "${EXPERIMENT}/results/feature_increment" \
  --input_audit "${EXPERIMENT}/results/ct_input_audit" \
  --results_root "${EXPERIMENT}/results/ct_phenotype_increment" \
  --workers 4 \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started CT phenotype experiment1 (CPU only), PID: $!"
echo "Log: ${LOG_FILE}"
