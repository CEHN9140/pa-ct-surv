BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/09_cptac_luad_pathology
EAGLE_ROOT=/home/gly001/cqj/EAGLE/test/cptac_luad

FEATURE_SOURCE=eagle_top25
PA_MODEL=meanpool_linear
NUM_EPOCHS=30
COX_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="cptac_luad_single_${FEATURE_SOURCE}_${PA_MODEL}_cox_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --eagle_root "${EAGLE_ROOT}" \
  --feature_source "${FEATURE_SOURCE}" \
  --pa_model "${PA_MODEL}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
