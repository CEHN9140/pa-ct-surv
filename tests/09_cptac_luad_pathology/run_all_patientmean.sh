BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/09_cptac_luad_pathology

COX_REDUCTION=patients
DROPOUT=0.25
NUM_EPOCHS=30
PATIENCE=10
COX_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=5e-4

mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/run_all.py" \
  --cox_reduction "${COX_REDUCTION}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --patience "${PATIENCE}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/run_all_patientmean.log" 2>&1 &

echo "PID: $!"
echo "Progress: ${EXPERIMENT}/logs/run_all_patientmean.log"
