BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/fold_difficulty_analysis/subbag80_stability"
if [ -e "${EXPERIMENT}/private" ]; then
    echo 'Existing diagnostic outputs; refusing overwrite' >&2
    exit 1
fi
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
setsid nohup "${BASE}/.radiomics_env/bin/python" -u "${EXPERIMENT}/pipeline.py" \
 > "${EXPERIMENT}/validation/pipeline.log" 2>&1 < /dev/null &
echo "Subbag diagnostic PID: $!"
