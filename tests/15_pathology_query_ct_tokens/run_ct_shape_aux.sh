BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUNTIME="${EXPERIMENT}/.phenotype_runtime/bin/python"

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${RUNTIME}" ]]; then
  echo "Missing isolated phenotype runtime; see README for setup"
  exit 1
fi
for TARGET in sphericity allshape; do
  if [[ -e "${EXPERIMENT}/logs/ct_shape_${TARGET}.log" || -e "${EXPERIMENT}/results/ct_shape_aux/${TARGET}" || -e "${EXPERIMENT}/checkpoints/ct_shape_aux/${TARGET}" ]]; then
    echo "CT auxiliary ${TARGET} already exists; refusing to overwrite"
    exit 1
  fi
done

for TARGET in sphericity allshape; do
  LOG_FILE="${EXPERIMENT}/logs/ct_shape_${TARGET}.log"
  MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 nohup setsid "${RUNTIME}" -u "${EXPERIMENT}/ct_shape_aux_train.py" \
    --target "${TARGET}" \
    --source_run "${EXPERIMENT}/results/nested_risk_increment" \
    --feature_run "${EXPERIMENT}/results/feature_increment" \
    --checkpoint_root "${EXPERIMENT}/checkpoints/ct_shape_aux/${TARGET}" \
    --results_root "${EXPERIMENT}/results/ct_shape_aux/${TARGET}" \
    --num_workers 8 \
    > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started CT auxiliary ${TARGET}, PID: $!"
  echo "Log: ${LOG_FILE}"
done
