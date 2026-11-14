BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUNTIME="${EXPERIMENT}/.phenotype_runtime/bin/python"
LOG="${EXPERIMENT}/logs/shape_risk_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${RUNTIME}" ]]; then
  echo "Missing isolated phenotype runtime; see README"
  exit 1
fi
if [[ -e "${LOG}" || -e "${EXPERIMENT}/results/shape_risk_increment" || -e "${EXPERIMENT}/checkpoints/shape_risk_increment" ]]; then
  echo "Shape risk increment output already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 \
nohup setsid "${RUNTIME}" -u "${EXPERIMENT}/shape_risk_increment.py" \
  --shape_weight 0.2 \
  --cox_ridge 1.0 \
  --num_workers 8 \
  > "${LOG}" 2>&1 < /dev/null &

echo "Started test15 fixed-Shape conditional risk diagnostic, PID: $!"
echo "Log: ${LOG}"
