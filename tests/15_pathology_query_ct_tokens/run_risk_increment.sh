BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
LOG_FILE="${EXPERIMENT}/logs/nested_risk_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/nested_risk_increment" || -e "${EXPERIMENT}/checkpoints/nested_risk_increment" ]]; then
  echo "Nested risk-increment experiment already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/risk_increment.py" \
  --data_dir "${BASE}/data/eagle/top25" \
  --ct_pretrained_path "${BASE}/model/ct_pretrain/resnet_18_23dataset.pth" \
  --inner_folds 3 \
  --stop_fraction 0.15 \
  --num_epochs 30 \
  --patience 10 \
  --cox_batch_size 32 \
  --batch_size 32 \
  --dropout 0.25 \
  --lr 1e-4 \
  --ct_backbone_lr 1e-5 \
  --weight_decay 5e-4 \
  --checkpoint_root "${EXPERIMENT}/checkpoints/nested_risk_increment" \
  --results_root "${EXPERIMENT}/results/nested_risk_increment" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started experiment 15 nested risk-increment diagnostic, PID: $!"
echo "Log: ${LOG_FILE}"
