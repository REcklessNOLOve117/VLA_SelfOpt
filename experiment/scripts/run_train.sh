#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${WAN_WM_PATH:?Set WAN_WM_PATH}"
: "${POC_RUN_DIR:?Set POC_RUN_DIR}"
: "${POC_NUM_NODES:?Set POC_NUM_NODES from topology_decision.json}"

export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
mkdir -p "${POC_RUN_DIR}"
python "${POC_PROJECT_ROOT}/scripts/run_with_budget.py" \
  --hours 72 \
  --grace-seconds 900 \
  --run-dir "${POC_RUN_DIR}" \
  -- python "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py" \
    --config-path "${POC_PROJECT_ROOT}/configs" \
    --config-name wan_libero_spatial_grpo_openvlaoft_lora32 \
    cluster.num_nodes="${POC_NUM_NODES}" \
    runner.max_steps=100
