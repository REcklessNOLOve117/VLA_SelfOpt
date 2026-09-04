#!/usr/bin/env bash
set -euo pipefail

: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_OFT_ROOT:?Set OPENVLA_OFT_ROOT}"
: "${POC_EVAL_POLICY_PATH:?Set POC_EVAL_POLICY_PATH to the Base or merged FT checkpoint}"
: "${POC_MODEL_LABEL:?Set POC_MODEL_LABEL to base or ft}"
: "${POC_RESULTS_DIR:?Set POC_RESULTS_DIR}"
: "${POC_SHARD_INDEX:?Set POC_SHARD_INDEX}"
: "${POC_NUM_SHARDS:?Set POC_NUM_SHARDS}"
: "${POC_RENDERER_BACKEND:=osmesa}"
if [[ "${POC_RENDERER_BACKEND}" != "osmesa" && "${POC_RENDERER_BACKEND}" != "egl" ]]; then
  echo "POC_RENDERER_BACKEND must be osmesa or egl" >&2
  exit 2
fi
export MUJOCO_GL="${POC_RENDERER_BACKEND}"
export PYOPENGL_PLATFORM="${POC_RENDERER_BACKEND}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

output_path="${POC_RESULTS_DIR}/${POC_MODEL_LABEL}-shard-${POC_SHARD_INDEX}.jsonl"
crash_retries=0
while true; do
  before_rows=0
  if [[ -f "${output_path}" ]]; then
    before_rows="$(wc -l < "${output_path}")"
  fi
  set +e
  python "${POC_PROJECT_ROOT}/scripts/eval_libero_manifest.py" \
    --policy-path "${POC_EVAL_POLICY_PATH}" \
    --model-label "${POC_MODEL_LABEL}" \
    --manifest "${POC_PROJECT_ROOT}/protocol/eval_manifest.jsonl" \
    --output "${output_path}" \
    --video-dir "${POC_RESULTS_DIR}/media/paired" \
    --shard-index "${POC_SHARD_INDEX}" \
    --num-shards "${POC_NUM_SHARDS}" \
    --max-action-steps 512 \
    --temperature 1.6 \
    --dtype float32
  return_code=$?
  set -e
  if [[ "${return_code}" -eq 0 ]]; then
    exit 0
  fi
  if [[ "${return_code}" -ne 134 && "${return_code}" -ne 139 ]]; then
    exit "${return_code}"
  fi
  after_rows=0
  if [[ -f "${output_path}" ]]; then
    after_rows="$(wc -l < "${output_path}")"
  fi
  if [[ "${after_rows}" -gt "${before_rows}" ]]; then
    crash_retries=1
  else
    crash_retries=$((crash_retries + 1))
  fi
  if [[ "${crash_retries}" -gt 2 ]]; then
    echo "Native evaluator crash repeated more than two times for the same pending episode" >&2
    exit "${return_code}"
  fi
  echo "Native evaluator crash rc=${return_code}; retry ${crash_retries}/2 from durable output" >&2
  sleep 5
done
