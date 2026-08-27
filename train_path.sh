BASE=/home/gly001/cqj/pa_ct_surv

ROI_SIZE=64
PA_MODEL=gabmil
DROPOUT=0.25
NUM_EPOCHS=30
LR=1e-4
WEIGHT_DECAY=5e-4
COX_BATCH_SIZE=32
SEED=123

RUN_TAG="roi${ROI_SIZE}_${PA_MODEL}_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}_seed${SEED}"

mkdir -p "${BASE}/logs/pact_v4/pathology"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python path_train.py \
  --ct_roi_size "${ROI_SIZE}" \
  --pa_model "${PA_MODEL}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --num_workers 8 \
  --patience 10 \
  --seed "${SEED}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v4/pathology/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v4/pathology/${RUN_TAG}" \
  > "${BASE}/logs/pact_v4/pathology/${RUN_TAG}.log" 2>&1 &