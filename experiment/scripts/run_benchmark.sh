#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${WAN_WM_PATH:?Set WAN_WM_PATH}"
: "${POC_RUN_DIR:?Set POC_RUN_DIR}"
: "${POC_NUM_NODES:?Set POC_NUM_NODES to 1 or 2}"
: "${POC_BENCHMARK_LABEL:?Set POC_BENCHMARK_LABEL}"
: "${NCCL_NVLS_ENABLE:?Set NCCL_NVLS_ENABLE=0 after the host NCCL smoke test}"
if [[ "${NCCL_NVLS_ENABLE}" != "0" ]]; then
  echo "NCCL_NVLS_ENABLE must be 0 for the verified POC runtime" >&2
  exit 2
fi

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
mkdir -p "${POC_RUN_DIR}"
python "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${POC_PROJECT_ROOT}/configs" \
  --config-name wan_libero_spatial_grpo_openvlaoft_lora32 \
  cluster.num_nodes="${POC_NUM_NODES}" \
  runner.max_steps=6 \
  runner.save_interval=1 \
  runner.logger.experiment_name="benchmark-${POC_BENCHMARK_LABEL}" \
  2>&1 | tee "${POC_RUN_DIR}/benchmark.log"

python "${POC_PROJECT_ROOT}/scripts/measure_benchmark.py" \
  --run-dir "${POC_RUN_DIR}" \
  --node "${POC_BENCHMARK_LABEL}" \
  --gpus "$((POC_NUM_NODES * 8))" \
  --output "${POC_RUN_DIR}/benchmark-result.json"
