BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/06_pathology_survival_losses

ROI_SIZE=64
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=64
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=5e-4

mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/run_all.py" \
  --ct_roi_size "${ROI_SIZE}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/run_all.log" 2>&1 &

echo "PID: $!"
echo "Progress: ${EXPERIMENT}/logs/run_all.log"
