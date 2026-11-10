BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/05_eagle_top25_pathology

PA_MODEL=meanpool
ROI_SIZE=64
DROPOUT=0.25
NUM_EPOCHS=30
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="eagle_top25_roi${ROI_SIZE}_${PA_MODEL}_cox_dropout${DROPOUT}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --data_dir "${BASE}/data/eagle/top25" \
  --pa_model "${PA_MODEL}" \
  --ct_roi_size "${ROI_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
