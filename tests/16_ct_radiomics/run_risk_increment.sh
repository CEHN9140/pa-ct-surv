BASE=/home/gly001/cqj/pa_ct_surv
EXPERIMENT="${BASE}/tests/16_ct_radiomics"
PYTHON="${EXPERIMENT}/.runtime/bin/python"
LOG="${EXPERIMENT}/logs/radiomics_risk_increment.log"

mkdir -p "${EXPERIMENT}/logs"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing experiment16 runtime; see ${EXPERIMENT}/README.md"
  exit 1
fi
if [[ -e "${LOG}" || -e "${EXPERIMENT}/conditional_risk_increment" ]]; then
  echo "Conditional increment log/results already exist; refusing to overwrite"
  exit 1
fi

MPLCONFIGDIR=/tmp/pact_matplotlib OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
setsid nohup "${PYTHON}" -u "${EXPERIMENT}/radiomics_risk_increment.py" \
  > "${LOG}" 2>&1 < /dev/null &
echo "Radiomics conditional risk increment started (CPU), PID: $!"
echo "Log: ${LOG}"
