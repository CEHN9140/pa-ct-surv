BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT=${BASE}/tests/10_pathology_learning_curve

MAX_UPDATES=180
EVAL_EVERY=9
COX_BATCH_SIZE=64
REPEATS=3
DROPOUT=0.25
LR=1e-4
WEIGHT_DECAY=5e-4

mkdir -p "${EXPERIMENT}/logs"

CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/train.py" \
  --max_updates "${MAX_UPDATES}" \
  --eval_every "${EVAL_EVERY}" \
  --cox_batch_size "${COX_BATCH_SIZE}" \
  --repeats "${REPEATS}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  >> "${EXPERIMENT}/logs/learning_curve.log" 2>&1 &

echo "PID: $!"
echo "Log: ${EXPERIMENT}/logs/learning_curve.log"
