BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/08_gl_pathology
DATA_DIR=${EXPERIMENT}/data_154_eventtime

SURVIVAL_LOSS=cox
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="gl154_eventtime_uni_v1_l2_abmil_${SURVIVAL_LOSS}_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --data_dir "${DATA_DIR}" \
  --l2_normalize \
  --survival_loss "${SURVIVAL_LOSS}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
