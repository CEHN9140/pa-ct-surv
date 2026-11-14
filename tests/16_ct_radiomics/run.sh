BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/16_ct_radiomics"
PYTHON="${EXPERIMENT}/.runtime/bin/python"
LOG="${EXPERIMENT}/logs/radiomics_cox.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing experiment16 runtime; see ${EXPERIMENT}/README.md"
  exit 1
fi
if [[ -e "${LOG}" || -e "${EXPERIMENT}/output/results" ]]; then
  echo "An experiment16 log/results directory already exists; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1 \
nohup "${PYTHON}" -u "${EXPERIMENT}/train.py" \
  --cohort_csv "${BASE}/data/eagle/top25/all_label_roi64.csv" \
  --ct_root /home/gly001/cqj/data/lung_cancer/ct \
  --output_dir "${EXPERIMENT}/output" \
  --workers 2 \
  > "${LOG}" 2>&1 < /dev/null &

echo "Experiment16 started (CPU only), PID: $!"
echo "Log: ${LOG}"
