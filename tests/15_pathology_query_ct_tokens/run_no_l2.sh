BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/15_pathology_query_ct_tokens"
RUN_TAG=eagle_top25_virchow2_cls_paraw_fusionraw_roi64_paq25_ctlayer4kv512_crossattn_attnpool_concat_dropout0.25_epochs30_coxbs32_lr1e-4_blr1e-5_wd5e-4
LOG_FILE="${EXPERIMENT}/logs/${RUN_TAG}.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ -e "${LOG_FILE}" || -e "${EXPERIMENT}/results/${RUN_TAG}" || -e "${EXPERIMENT}/checkpoints/${RUN_TAG}" ]]; then
  echo "Experiment already exists; refusing to overwrite ${RUN_TAG}"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib CUDA_VISIBLE_DEVICES=0 nohup /home/gly001/.conda/envs/UNI/bin/python -u "${EXPERIMENT}/run.py" \
  --data_dir "${BASE}/data/eagle/top25" \
  --ct_pretrained_path "${BASE}/model/ct_pretrain/resnet_18_23dataset.pth" \
  --no_pa_l2_normalize \
  --alignment_weight 0 \
  --num_epochs 30 \
  --cox_batch_size 32 \
  --dropout 0.25 \
  --lr 1e-4 \
  --ct_backbone_lr 1e-5 \
  --weight_decay 5e-4 \
  --patience 10 \
  --checkpoint_root "${EXPERIMENT}/checkpoints/${RUN_TAG}" \
  --results_root "${EXPERIMENT}/results/${RUN_TAG}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Started experiment 15 without pathology L2, PID: $!"
echo "Log: ${LOG_FILE}"
