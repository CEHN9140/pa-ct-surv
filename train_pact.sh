BASE=/home/gly001/cqj/pa_ct_surv

ROI_SIZE=64
PA_MODEL=abmil
CT_MODEL=resnet18
FUSION_TYPE=concat
NORM=layernorm
CT_PRETRAINED_PATH=${BASE}/model/ct_pretrain/resnet_18_23dataset.pth

NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
LAMBDA_CT=0.0
LAMBDA_PA=0.0
FUSION_DROPOUT=0.5
COX_BATCH_SIZE=64
SEED=42

RUN_TAG="roi${ROI_SIZE}_${PA_MODEL}_${CT_MODEL}_${FUSION_TYPE}_norm${NORM}_fusion_dropout${FUSION_DROPOUT}_epochs${NUM_EPOCHS}_coxbs${COX_BATCH_SIZE}_lr${LR}_blr${BACKBONE_LR}_wd${WEIGHT_DECAY}_lct${LAMBDA_CT}_lpa${LAMBDA_PA}_seed${SEED}"

mkdir -p "${BASE}/logs/pact_v4/pact"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python pact_train.py \
  --ct_roi_size "${ROI_SIZE}" \
  --pa_model "${PA_MODEL}" \
  --ct_model "${CT_MODEL}" \
  --ct_pretrained_path "${CT_PRETRAINED_PATH}" \
  --fusion_type "${FUSION_TYPE}" \
  --norm "${NORM}" \
  --fusion_dropout "${FUSION_DROPOUT}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --ct_backbone_lr "${BACKBONE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --lambda_ct "${LAMBDA_CT}" \
  --lambda_pa "${LAMBDA_PA}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --num_workers 8 \
  --patience 10 \
  --seed "${SEED}" \
  --checkpoint_root "${BASE}/checkpoints/pact_v4/pact/${RUN_TAG}" \
  --results_root "${BASE}/results/pact_v4/pact/${RUN_TAG}" \
  > "${BASE}/logs/pact_v4/pact/${RUN_TAG}.log" 2>&1 &
