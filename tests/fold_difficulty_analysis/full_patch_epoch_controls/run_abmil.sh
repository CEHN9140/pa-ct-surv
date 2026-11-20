BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/fold_difficulty_analysis/full_patch_epoch_controls/abmil"
if [ -e "${EXPERIMENT}/training.log" ] || [ -e "${EXPERIMENT}/results" ] || [ -e "${EXPERIMENT}/checkpoints" ]; then
    echo "Existing abmil run; refusing to overwrite." >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLBACKEND=Agg MPLCONFIGDIR="${EXPERIMENT}/matplotlib" \
setsid nohup "${BASE}/.radiomics_env/bin/python" -u "${BASE}/path_train.py" \
  --data_dir "${BASE}/data/trident/features_uni_v1" --ct_roi_size 64 --feature_dim 1024 \
  --pa_model abmil --survival_loss cox --dropout 0.25 \
  --num_epochs 30 --patience 10 --cox_batch_size 32 --lr 1e-4 --weight_decay 5e-4 \
  --num_workers 8 \
  --checkpoint_root "${EXPERIMENT}/checkpoints" --results_root "${EXPERIMENT}/results" \
  > "${EXPERIMENT}/training.log" 2>&1 < /dev/null &
echo "abmil PID: $!"
echo "Log: ${EXPERIMENT}/training.log"
