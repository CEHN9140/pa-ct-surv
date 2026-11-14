BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/16_ct_radiomics"
PYTHON="${EXPERIMENT}/.runtime/bin/python"
DROPOUT=0.25
COX_BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=5e-4
NUM_EPOCHS=30
PATIENCE=10

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing experiment16 runtime; see ${EXPERIMENT}/README.md"
  exit 1
fi
for MODEL in concat pa_only; do
  if [[ -e "${EXPERIMENT}/logs/teacher_${MODEL}.log" || -e "${EXPERIMENT}/teacher_results/${MODEL}" || -e "${EXPERIMENT}/teacher_checkpoints/${MODEL}" ]]; then
    echo "Existing ${MODEL} run found; refusing to overwrite"
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
setsid nohup "${PYTHON}" -u "${EXPERIMENT}/teacher_train.py" \
  --fusion_type concat \
  --dropout "${DROPOUT}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" \
  --num_epochs "${NUM_EPOCHS}" --patience "${PATIENCE}" \
  > "${EXPERIMENT}/logs/teacher_concat.log" 2>&1 < /dev/null &
echo "Concat PID: $!"

CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
setsid nohup "${PYTHON}" -u "${EXPERIMENT}/teacher_train.py" \
  --fusion_type pa_only \
  --dropout "${DROPOUT}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" \
  --num_epochs "${NUM_EPOCHS}" --patience "${PATIENCE}" \
  > "${EXPERIMENT}/logs/teacher_pa_only.log" 2>&1 < /dev/null &
echo "Matched PA-only PID: $!"
echo "Logs: ${EXPERIMENT}/logs/teacher_concat.log and teacher_pa_only.log"
