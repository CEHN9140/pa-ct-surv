BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"

mkdir -p "${EXPERIMENT}/logs"
for FEATURE_SOURCE in layer3 layer34; do
  if [[ -e "${EXPERIMENT}/logs/ct_${FEATURE_SOURCE}.log" || -e "${EXPERIMENT}/results/ct_layer_comparison/${FEATURE_SOURCE}" || -e "${EXPERIMENT}/checkpoints/ct_layer_comparison/${FEATURE_SOURCE}" ]]; then
    echo "CT ${FEATURE_SOURCE} already exists; refusing to overwrite"
    exit 1
  fi
done

for FEATURE_SOURCE in layer3 layer34; do
  LOG_FILE="${EXPERIMENT}/logs/ct_${FEATURE_SOURCE}.log"
  MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0 nohup setsid /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/ct_layer_train.py" \
    --feature_source "${FEATURE_SOURCE}" \
    --source_run "${EXPERIMENT}/results/nested_risk_increment" \
    --input_audit "${EXPERIMENT}/results/ct_input_audit" \
    --num_workers 8 \
    --checkpoint_root "${EXPERIMENT}/checkpoints/ct_layer_comparison/${FEATURE_SOURCE}" \
    --results_root "${EXPERIMENT}/results/ct_layer_comparison/${FEATURE_SOURCE}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started CT ${FEATURE_SOURCE}, PID: $!"
  echo "Log: ${LOG_FILE}"
done
