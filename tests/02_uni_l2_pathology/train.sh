BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/02_uni_l2_pathology

ROI_SIZE=64
PA_MODEL=abmil
DROPOUT=0.25
NUM_EPOCHS=30
COX_BATCH_SIZE=16
LR=1e-4
WEIGHT_DECAY=5e-4

RUN_TAG="uni_l2_roi${ROI_SIZE}_${PA_MODEL}_cox_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}_diagnostics_stability"
mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/repeat_seeds.py" \
  --data_dir "${BASE}/data/trident/features_uni_v1" \
  --ct_roi_size "${ROI_SIZE}" \
  --pa_model "${PA_MODEL}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  > "${EXPERIMENT}/logs/${RUN_TAG}.log" 2>&1 &

echo "Started sequential seed42 and seed2024 training, PID: $!"
echo "Progress: ${EXPERIMENT}/logs/${RUN_TAG}.log"
echo "Training logs: ${EXPERIMENT}/logs/stability_seed42.log and stability_seed2024.log"
