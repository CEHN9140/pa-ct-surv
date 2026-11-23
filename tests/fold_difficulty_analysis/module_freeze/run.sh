BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/fold_difficulty_analysis/module_freeze"
if [ -e "${EXPERIMENT}/status.json" ] || [ -e "${EXPERIMENT}/fold_0" ]; then
    echo 'Existing module-freeze experiment; refusing overwrite' >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg \
MPLCONFIGDIR="${EXPERIMENT}/validation/matplotlib" \
setsid nohup "${BASE}/.radiomics_env/bin/python" -u "${EXPERIMENT}/pipeline.py" \
 > "${EXPERIMENT}/validation/pipeline.log" 2>&1 < /dev/null &
echo "Module-freeze pipeline PID: $!"
