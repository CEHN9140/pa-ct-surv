BASE=/home/gly001/cqj/pa_ct_surv

ROI_SIZE=64
PA_MODEL=abmil
PCA_DIM=256
PCA_PATCHES_PER_PATIENT=256
ABMIL_HIDDEN_DIM=128
ABMIL_ATTENTION_DIM=32
DROPOUT=0.0
NUM_EPOCHS=60
LR=1e-4
WEIGHT_DECAY=5e-4
COX_BATCH_SIZE=64
SEED=42

RUN_TAG="roi${ROI_SIZE}_${PA_MODEL}_pca${PCA_DIM}_ppp${PCA_PATCHES_PER_PATIENT}_hd${ABMIL_HIDDEN_DIM}_ad${ABMIL_ATTENTION_DIM}_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}_seed${SEED}"

PCA_ARGS=()
if [ "${PCA_DIM}" != "none" ]; then
  PCA_ARGS+=(--pca_dim "${PCA_DIM}")
fi
 
mkdir -p "${BASE}/logs/pact_v5/pathology"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python path_train.py \
  --ct_roi_size "${ROI_SIZE}" \
  --pa_model "${PA_MODEL}" \
  "${PCA_ARGS[@]}" \
  --pca_patches_per_patient "${PCA_PATCHES_PER_PATIENT}" \
  --abmil_hidden_dim "${ABMIL_HIDDEN_DIM}" \
  --abmil_attention_dim "${ABMIL_ATTENTION_DIM}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --num_workers 8 \
  --patience 10 \
  --seed "${SEED}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v5/pathology/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v5/pathology/${RUN_TAG}" \
  > "${BASE}/logs/pact_v5/pathology/${RUN_TAG}.log" 2>&1 &
