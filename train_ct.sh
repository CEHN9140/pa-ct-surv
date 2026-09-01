BASE=/home/gly001/cqj/pa_ct_surv

ROI_SIZE=128
CT_MODEL=resnet18
CT_PRETRAINED_PATH=${BASE}/model/ct_pretrain/resnet_18_23dataset.pth
DROPOUT=0.5
NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
BATCH_SIZE=64
SEED=42

RUN_TAG="roi${ROI_SIZE}_${CT_MODEL}_dropout${DROPOUT}_epochs${NUM_EPOCHS}_bs${BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_wd${WEIGHT_DECAY}_seed${SEED}"

mkdir -p "${BASE}/logs/pact_v4/ct"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python ct_train.py \
  --ct_roi_size "${ROI_SIZE}" \
  --ct_model "${CT_MODEL}" \
  --ct_pretrained_path "${CT_PRETRAINED_PATH}" \
  --dropout "${DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --ct_backbone_lr "${BACKBONE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers 8 \
  --patience 10 \
  --seed "${SEED}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v4/ct/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v4/ct/${RUN_TAG}" \
  > "${BASE}/logs/pact_v4/ct/${RUN_TAG}.log" 2>&1 &