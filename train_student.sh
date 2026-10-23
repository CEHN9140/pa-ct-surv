BASE=/home/gly001/cqj/pa_ct_surv
DATA_SOURCE=trident
FEATURE_NAME=features_uni_v1
DATA_DIR="${BASE}/data/${DATA_SOURCE}/${FEATURE_NAME}"

STUDENT_MODEL=resnet18
STUDENT_PRETRAINED_PATH=${BASE}/model/ct_pretrain/resnet_18_23dataset.pth
STUDENT_DROPOUT=0.0
TEACHER_TAG="roi64_abmil_resnet18_weighted_normnone_nodropout_epochs30_coxbs64_lr1e-4_blr1e-5_wd5e-4_lct0.5_lpa0.3"
TEACHER_CKPT_ROOT="${BASE}/checkpoints/pact_v6/pact/${TEACHER_TAG}"

DISTILL_MODE=mse
ALPHA_KD=0.3
KD_TEMPERATURE=1.0

NUM_EPOCHS=30
LR=1e-4
BACKBONE_LR=1e-5
WEIGHT_DECAY=5e-4
COX_BATCH_SIZE=64

RUN_TAG="${TEACHER_TAG}_${STUDENT_MODEL}_dropout${STUDENT_DROPOUT}_${DISTILL_MODE}_akd${ALPHA_KD}_t${KD_TEMPERATURE}"
CHECKPOINT_ROOT="${BASE}/checkpoints/pact_v6/ct_student/${RUN_TAG}"
RESULTS_ROOT="${BASE}/results/pact_v6/ct_student/${RUN_TAG}"
LOG_ROOT="${BASE}/logs/pact_v6/ct_student"

mkdir -p "${LOG_ROOT}"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python distill_train.py \
  --data_dir "${DATA_DIR}" \
  --student_model "${STUDENT_MODEL}" \
  --student_pretrained_path "${STUDENT_PRETRAINED_PATH}" \
  --student_dropout "${STUDENT_DROPOUT}" \
  --teacher_ckpt_root "${TEACHER_CKPT_ROOT}" \
  --distill_mode "${DISTILL_MODE}" \
  --alpha_kd "${ALPHA_KD}" \
  --kd_temperature "${KD_TEMPERATURE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --ct_backbone_lr "${BACKBONE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --results_root "${RESULTS_ROOT}" \
  > "${LOG_ROOT}/${RUN_TAG}.log" 2>&1 &
