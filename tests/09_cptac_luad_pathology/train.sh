BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/09_cptac_luad_pathology

FEATURE_SOURCE=uni
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="cptac_luad_single_${FEATURE_SOURCE}_abmil_cox_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --feature_source "${FEATURE_SOURCE}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
