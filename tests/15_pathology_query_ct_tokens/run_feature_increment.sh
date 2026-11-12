BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
LOG_FILE="${EXPERIMENT}/logs/feature_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/feature_increment" || -e "${EXPERIMENT}/checkpoints/feature_increment" ]]; then
  echo "Feature-increment diagnostic already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/feature_increment.py" \
  --source_run "${EXPERIMENT}/results/nested_risk_increment" \
  --results_root "${EXPERIMENT}/results/feature_increment" \
  --checkpoint_root "${EXPERIMENT}/checkpoints/feature_increment" \
  --ct_sources public finetuned \
  --pca_dims 16 32 \
  --ridge_grid 0.01 0.1 1 10 100 1000 \
  --num_epochs 30 \
  --patience 10 \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started experiment15 CT-feature increment and conditional Shape diagnostic, PID: $!"
echo "Log: ${LOG_FILE}"
