BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/16_ct_radiomics"
PYTHON="${EXPERIMENT}/.runtime/bin/python"
FUSION_TYPE=crossattention
DROPOUT=0.25
COX_BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=5e-4
NUM_EPOCHS=30
PATIENCE=10
LOG="${EXPERIMENT}/logs/teacher_${FUSION_TYPE}.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing experiment16 runtime; see ${EXPERIMENT}/README.md"
  exit 1
fi
if [[ -e "${LOG}" || -e "${EXPERIMENT}/teacher_results/${FUSION_TYPE}" || -e "${EXPERIMENT}/teacher_checkpoints/${FUSION_TYPE}" ]]; then
  echo "Teacher log/results/checkpoints already exist; refusing to overwrite"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
nohup "${PYTHON}" -u "${EXPERIMENT}/teacher_train.py" \
  --fusion_type "${FUSION_TYPE}" \
  --dropout "${DROPOUT}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_epochs "${NUM_EPOCHS}" \
  --patience "${PATIENCE}" \
  > "${LOG}" 2>&1 < /dev/null &

echo "Experiment16 teacher started, PID: $!"
echo "Log: ${LOG}"
