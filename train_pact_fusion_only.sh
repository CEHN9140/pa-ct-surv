BASE=/home/gly001/cqj/pa_ct_surv
DATA_DIR="${BASE}/data/eagle/top25"

ROI_SIZE=64
PA_MODEL=abmil
CT_MODEL=resnet18
SURVIVAL_LOSS=cox
DROPOUT=0.25
CT_PRETRAINED_PATH="${BASE}/model/ct_pretrain/resnet_18_23dataset.pth"

NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
LAMBDA_CT=0
LAMBDA_PA=0
COX_BATCH_SIZE=32

mkdir -p "${BASE}/logs/pact_v7/pact"

# Launch all five fusion experiments concurrently; each runs its own five folds.
for FUSION_TYPE in concat weighted gated bilinear crossattention; do
  RUN_TAG="eagle_top25_virchow2_cls_pal2_fusionl2_sharedct_cv5_roi${ROI_SIZE}_${PA_MODEL}_${CT_MODEL}_${FUSION_TYPE}_dim256_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_wd${WEIGHT_DECAY}_lct${LAMBDA_CT}_lpa${LAMBDA_PA}_fusiononly"
  LOG_FILE="${BASE}/logs/pact_v7/pact/${RUN_TAG}.log"
  CHECKPOINT_ROOT="${BASE}/checkpoints/pact_v7/pact/${RUN_TAG}"
  RESULTS_ROOT="${BASE}/results/pact_v7/pact/${RUN_TAG}"

  if [[ -e "${LOG_FILE}" || -e "${CHECKPOINT_ROOT}" || -e "${RESULTS_ROOT}" ]]; then
    echo "Skipping existing experiment without overwriting: ${RUN_TAG}"
    continue
  fi

  echo "Starting ${FUSION_TYPE}: $(date '+%F %T')"
  echo "Log: ${LOG_FILE}"
  CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${BASE}/pact_train.py" \
    --data_dir "${DATA_DIR}" \
    --ct_roi_size "${ROI_SIZE}" \
    --pa_model "${PA_MODEL}" \
    --pa_feature_dim 1280 \
    --pa_l2_normalize \
    --fusion_l2_normalize \
    --shared_ct_feature \
    --ct_model "${CT_MODEL}" \
    --ct_pretrained_path "${CT_PRETRAINED_PATH}" \
    --fusion_type "${FUSION_TYPE}" \
    --survival_loss "${SURVIVAL_LOSS}" \
    --dropout "${DROPOUT}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --ct_backbone_lr "${BACKBONE_LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --lambda_ct "${LAMBDA_CT}" \
    --lambda_pa "${LAMBDA_PA}" \
    --cox_batch_size "${COX_BATCH_SIZE}" \
    --checkpoint_root "${CHECKPOINT_ROOT}" \
    --results_root "${RESULTS_ROOT}" \
    > "${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started ${FUSION_TYPE}, PID: $!"
done

echo "Parallel launch finished; inspect each experiment log for training progress."
