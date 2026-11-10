BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/04_random_init_ct

ROI_SIZE=96
CT_MODEL=resnet18
DROPOUT=0.25
NUM_EPOCHS=100
BATCH_SIZE=64
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="roi${ROI_SIZE}_${CT_MODEL}_randominit_bntrain_dropout${DROPOUT}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --data_dir "${BASE}/data/trident/features_uni_v1" \
  --ct_roi_size "${ROI_SIZE}" \
  --ct_model "${CT_MODEL}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "Started experiment 04, PID: $!"
echo "Log: ${EXPERIMENT}/logs/${RUN_TAG}.log"
