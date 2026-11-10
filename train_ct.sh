BASE=/home/gly001/cqj/pa_ct_surv
DATA_SOURCE=trident
FEATURE_NAME=features_uni_v1
DATA_DIR="${BASE}/data/${DATA_SOURCE}/${FEATURE_NAME}"

ROI_SIZE=64
CT_MODEL=resnet18
SURVIVAL_LOSS=cox  # cox, cox_rank
CT_PRETRAINED_PATH=${BASE}/model/ct_pretrain/resnet_18_23dataset.pth
DROPOUT=0.25
NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
BATCH_SIZE=32

RUN_TAG="roi${ROI_SIZE}_${CT_MODEL}_l2_dropout${DROPOUT}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_wd${WEIGHT_DECAY}"
if [[ "${SURVIVAL_LOSS}" == "cox_rank" ]]; then
  RUN_TAG="${RUN_TAG}_cox_rank"
fi

mkdir -p "${BASE}/logs/pact_v7/ct"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${BASE}/ct_train.py" \
  --data_dir "${DATA_DIR}" \
  --ct_roi_size "${ROI_SIZE}" \
  --ct_model "${CT_MODEL}" \
  --survival_loss "${SURVIVAL_LOSS}" \
  --ct_pretrained_path "${CT_PRETRAINED_PATH}" \
  --dropout "${DROPOUT}" \
  --l2_normalize \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --ct_backbone_lr "${BACKBONE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v7/ct/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v7/ct/${RUN_TAG}" \
  > "${BASE}/logs/pact_v7/ct/${RUN_TAG}.log" 2>&1 &
