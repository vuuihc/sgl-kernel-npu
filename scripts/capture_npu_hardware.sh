#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR=${1:?usage: capture_npu_hardware.sh RESULT_DIR}
mkdir -p "${RESULT_DIR}"

capture_optional() {
    local output_file=$1
    shift
    {
        echo "\$ $*"
        "$@"
    } >"${RESULT_DIR}/${output_file}" 2>&1 || true
}

capture_optional npu-smi-info.txt npu-smi info
capture_optional npu-smi-board.txt npu-smi info -t board
capture_optional npu-smi-product.txt npu-smi info -t product
capture_optional npu-smi-version.txt npu-smi info -t health
capture_optional uname.txt uname -a
capture_optional lscpu.txt lscpu

{
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "python=$(python3 --version 2>&1 || true)"
    echo "torch=$(python3 -c 'import torch; print(torch.__version__)' 2>&1 || true)"
    echo "torch_npu=$(python3 -c 'import torch_npu; print(torch_npu.__version__)' 2>&1 || true)"
    echo "soc_version=$(python3 -c 'import torch_npu; print(torch_npu.npu._backends.get_soc_version())' 2>&1 || true)"
    echo "device_count=$(python3 -c 'import torch; print(torch.npu.device_count())' 2>&1 || true)"
    echo "device_name=$(python3 -c 'import torch; print(torch.npu.get_device_name())' 2>&1 || true)"
} >"${RESULT_DIR}/software-and-device.txt"

if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    {
        echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
        echo "ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME:-}"
        echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH:-}"
    } >"${RESULT_DIR}/ascend-environment.txt"
else
    env | grep -E '^(ASCEND|CANN|PATH|LD_LIBRARY_PATH)=' \
        >"${RESULT_DIR}/ascend-environment.txt" || true
fi

echo "Hardware and software snapshot written to ${RESULT_DIR}"
