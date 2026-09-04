#!/usr/bin/env bash
set -euo pipefail

: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${POC_RUN_DIR:?Set POC_RUN_DIR}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${WAN_WM_PATH:?Set WAN_WM_PATH}"
: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${POC_CANARY_OWNS_RAY:?Set POC_CANARY_OWNS_RAY=1 only for an exclusive canary cluster}"
test "${POC_CANARY_OWNS_RAY}" = "1"
mkdir -p "${POC_RUN_DIR}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
trap 'ray stop --force >/dev/null 2>&1 || true' EXIT

python "${POC_PROJECT_ROOT}/scripts/gpu_cleanup_check.py" snapshot --output "${POC_RUN_DIR}/gpu-before.json"
CANARY_ARGS=(
  --config-dir "${POC_PROJECT_ROOT}/configs"
  --policy-path "${OPENVLA_BASE_PATH}"
  --output "${POC_RUN_DIR}/rollout"
  --chunks "${POC_CANARY_CHUNKS:-32}"
)
if [[ -n "${POC_CANARY_RECORD_NAME:-}" ]]; then
  CANARY_ARGS+=(--record-name "${POC_CANARY_RECORD_NAME}")
fi
python "${POC_PROJECT_ROOT}/scripts/wan_openvla_canary.py" "${CANARY_ARGS[@]}"

bash "${POC_PROJECT_ROOT}/scripts/run_grpo_canary.sh"

SECOND_CHECKPOINT="$(find "${POC_RUN_DIR}" -type f -path '*global_step_2*' -name '*.pt' | head -n 1)"
test -n "${SECOND_CHECKPOINT}"
export POC_FSDP_CHECKPOINT="${SECOND_CHECKPOINT}"
export POC_EXPORT_DIR="${POC_RUN_DIR}/export"
bash "${POC_PROJECT_ROOT}/scripts/export_checkpoint.sh"

python "${POC_PROJECT_ROOT}/scripts/compare_merged_actions.py" \
  --base-policy "${OPENVLA_BASE_PATH}" \
  --adapter "${POC_EXPORT_DIR}/adapter/lora_adapter" \
  --merged-policy "${POC_EXPORT_DIR}/merged" \
  --wan-dataset "${WAN_WM_PATH}/dataset" \
  --output "${POC_RUN_DIR}/merge-report.json"

ray stop --force
trap - EXIT
python "${POC_PROJECT_ROOT}/scripts/gpu_cleanup_check.py" check \
  --before "${POC_RUN_DIR}/gpu-before.json" \
  --output "${POC_RUN_DIR}/cleanup-report.json"

python "${POC_PROJECT_ROOT}/scripts/build_training_canary_report.py" \
  --run-dir "${POC_RUN_DIR}" \
  --audit "${POC_RUN_DIR}/lora-audit.json" \
  --hash-before "${POC_RUN_DIR}/assets-before.json" \
  --hash-after "${POC_RUN_DIR}/assets-after.json" \
  --merge-report "${POC_RUN_DIR}/merge-report.json" \
  --output "${POC_RUN_DIR}/training-report.json"

python "${POC_PROJECT_ROOT}/scripts/validate_canary.py" \
  "${POC_RUN_DIR}/rollout/rollout_report.json" \
  "${POC_RUN_DIR}/training-report.json" \
  "${POC_RUN_DIR}/cleanup-report.json" \
  --output "${POC_RUN_DIR}/canary-acceptance.json"
