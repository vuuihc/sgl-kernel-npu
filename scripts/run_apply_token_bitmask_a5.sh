#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
SUITE=${1:-full}
SOC_VERSION=${SOC_VERSION:-Ascend950PR_9599}
RESULT_DIR=${2:-"${ROOT_DIR}/benchmark/results/apply_token_bitmask_a5/${TIMESTAMP}"}

if [[ "${SUITE}" != "quick" && "${SUITE}" != "full" ]]; then
    echo "Usage: $0 [quick|full] [result-directory]" >&2
    exit 2
fi

cd "${ROOT_DIR}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to benchmark a dirty worktree:" >&2
    git status --short >&2
    exit 1
fi

echo "Building sgl_kernel_npu for ${SOC_VERSION}..."
bash build.sh -a kernels "${SOC_VERSION}"

shopt -s nullglob
wheels=(output/sgl_kernel_npu*.whl)
if [[ ${#wheels[@]} -ne 1 ]]; then
    echo "Expected exactly one sgl_kernel_npu wheel, found ${#wheels[@]}" >&2
    exit 1
fi
python3 -m pip install --force-reinstall --no-deps "${wheels[0]}"

mkdir -p "${RESULT_DIR}"
echo "Capturing hardware and software metadata..."
bash scripts/capture_npu_hardware.sh "${RESULT_DIR}"

echo "Running functional regression tests..."
{
    python3 tests/python/sgl_kernel_npu/test_apply_token_bitmask.py --category boundary
    if [[ "${SUITE}" == "full" ]]; then
        python3 tests/python/sgl_kernel_npu/test_apply_token_bitmask.py --category llm
        python3 tests/python/sgl_kernel_npu/test_apply_token_bitmask.py --category general
    fi
} 2>&1 | tee "${RESULT_DIR}/functional-tests.txt"

npu-smi info >"${RESULT_DIR}/npu-smi.txt"
python3 benchmark/bench_apply_token_bitmask.py \
    --suite "${SUITE}" \
    --output "${RESULT_DIR}/results.json" \
    --markdown-output "${RESULT_DIR}/results.md"

git rev-parse HEAD >"${RESULT_DIR}/commit.txt"
git status --short >"${RESULT_DIR}/git-status.txt"
printf 'soc_version=%s\nsuite=%s\n' "${SOC_VERSION}" "${SUITE}" \
    >"${RESULT_DIR}/runner-config.txt"

echo
echo "Results are ready in ${RESULT_DIR}"
echo "Review results.md, then commit only this result directory on the experiment branch."
