BASE=/home/gly001/cqj/pa_ct_surv
DATA_SOURCE=eagle
FEATURE_NAME=top25
DATA_DIR="${BASE}/data/${DATA_SOURCE}/${FEATURE_NAME}"

ROI_SIZE=64
PA_MODEL=abmil
CT_MODEL=resnet18
FUSION_TYPE=crossattention  # concat, weighted, gated, bilinear, crossattention
SURVIVAL_LOSS=cox  # cox, cox_rank
DROPOUT=0.25
CT_PRETRAINED_PATH=${BASE}/model/ct_pretrain/resnet_18_23dataset.pth

NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
LAMBDA_CT=0.5
LAMBDA_PA=0.3
COX_BATCH_SIZE=32

RUN_TAG="${DATA_SOURCE}_${FEATURE_NAME}_virchow2_cls_pal2_fusionraw_sharedct_cv5_roi${ROI_SIZE}_${PA_MODEL}_${CT_MODEL}_${FUSION_TYPE}_dim256_dropout${DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_wd${WEIGHT_DECAY}_lct${LAMBDA_CT}_lpa${LAMBDA_PA}"
if [[ "${SURVIVAL_LOSS}" == "cox_rank" ]]; then
  RUN_TAG="${RUN_TAG}_cox_rank"
fi

mkdir -p "${BASE}/logs/pact_v7/pact"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${BASE}/pact_train.py" \
  --data_dir "${DATA_DIR}" \
  --ct_roi_size "${ROI_SIZE}" \
  --pa_model "${PA_MODEL}" \
  --pa_feature_dim 1280 \
  --pa_l2_normalize \
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
  --checkpoint_root "${BASE}/checkpoints/pact_v7/pact/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v7/pact/${RUN_TAG}" \
  > "${BASE}/logs/pact_v7/pact/${RUN_TAG}.log" 2>&1 &

echo "Started ${FUSION_TYPE}, PID: $!"
echo "Log: ${BASE}/logs/pact_v7/pact/${RUN_TAG}.log"
