#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${OPD_WORK_DIR:-/home/slu375/llm4cov_eda_work}"
SCRIPT="${WORK_DIR}/opd_worker.py"
LOG="${WORK_DIR}/opd_worker.log"
PID_FILE="${WORK_DIR}/opd_worker.pid"

export OPD_SLIME_DIR="${OPD_SLIME_DIR:-/home/slu375/docker_scripts/ctr_slime/slime-OPD}"
export PYTHONPATH="${OPD_SLIME_DIR}/third_party/llm4cov_oss/src:${PYTHONPATH:-}"

if pgrep -f "${SCRIPT}" >/dev/null 2>&1; then
  echo "opd_worker already running"
  pgrep -af "${SCRIPT}"
  exit 0
fi

mkdir -p "${WORK_DIR}"
nohup /usr/bin/python3 "${SCRIPT}" >>"${LOG}" 2>&1 &
echo "$!" >"${PID_FILE}"
echo "started opd_worker pid=$(cat "${PID_FILE}") log=${LOG}"
