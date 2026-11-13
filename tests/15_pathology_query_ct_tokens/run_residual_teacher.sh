BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
LOG_FILE="${EXPERIMENT}/logs/residual_teacher.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/residual_teacher" || -e "${EXPERIMENT}/checkpoints/residual_teacher" ]]; then
  echo "Residual teacher experiment already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup setsid /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/residual_train.py" \
  --source_run "${EXPERIMENT}/results/nested_risk_increment" \
  --correction_weights 0 0.01 \
  --lambda_pa 0.3 \
  --num_epochs 30 \
  --patience 10 \
  --cox_batch_size 32 \
  --dropout 0.25 \
  --lr 1e-4 \
  --ct_backbone_lr 1e-5 \
  --weight_decay 5e-4 \
  --num_workers 8 \
  --checkpoint_root "${EXPERIMENT}/checkpoints/residual_teacher" \
  --results_root "${EXPERIMENT}/results/residual_teacher" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started experiment15 residual teachers, PID: $!"
echo "Log: ${LOG_FILE}"
