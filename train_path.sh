BASE=/home/gly001/cqj/pa_ct_surv
DATA_SOURCE=eagle
FEATURE_NAME=top25

ROI_SIZE=64
PA_MODEL=abmil
SURVIVAL_LOSS=cox
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=5e-4
DATA_DIR="${BASE}/data/${DATA_SOURCE}/${FEATURE_NAME}"

RUN_TAG="${DATA_SOURCE}_${FEATURE_NAME}_virchow2_cls_raw_cv5_roi${ROI_SIZE}_${PA_MODEL}_${SURVIVAL_LOSS}_dropout${DROPOUT}_epochs${NUM_EPOCHS}"
if [[ "${SURVIVAL_LOSS}" != "nll" ]]; then
  RUN_TAG="${RUN_TAG}_coxbs${COX_BATCH_SIZE}"
fi
RUN_TAG="${RUN_TAG}_lr${LR}_wd${WEIGHT_DECAY}"
 
mkdir -p "${BASE}/logs/pact_v7/pathology"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${BASE}/path_train.py" \
  --data_dir "${DATA_DIR}" \
  --ct_roi_size "${ROI_SIZE}" \
  --feature_dim 1280 \
  --pa_model "${PA_MODEL}" \
  --survival_loss "${SURVIVAL_LOSS}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v7/pathology/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v7/pathology/${RUN_TAG}" \
  > "${BASE}/logs/pact_v7/pathology/${RUN_TAG}.log" 2>&1 &
