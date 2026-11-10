BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/06_pathology_survival_losses

FEATURE_SOURCE=uni_l2  # uni_l2, keep, eagle
SURVIVAL_LOSS=nll  # nll, cox_rank
ROI_SIZE=64
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=64
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="${FEATURE_SOURCE}_${SURVIVAL_LOSS}_roi${ROI_SIZE}_dropout${DROPOUT}_epochs${NUM_EPOCHS}"
if [[ "${FEATURE_SOURCE}" == "eagle" ]]; then
  RUN_TAG="${RUN_TAG}_bs${BATCH_SIZE}"
elif [[ "${SURVIVAL_LOSS}" == "cox_rank" ]]; then
  RUN_TAG="${RUN_TAG}_coxbs${COX_BATCH_SIZE}"
fi
RUN_TAG="${RUN_TAG}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --feature_source "${FEATURE_SOURCE}" \
  --survival_loss "${SURVIVAL_LOSS}" \
  --ct_roi_size "${ROI_SIZE}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
