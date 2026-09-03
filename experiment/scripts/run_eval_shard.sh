#!/usr/bin/env bash
set -euo pipefail

: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_OFT_ROOT:?Set OPENVLA_OFT_ROOT}"
: "${POC_EVAL_POLICY_PATH:?Set POC_EVAL_POLICY_PATH to the Base or merged FT checkpoint}"
: "${POC_MODEL_LABEL:?Set POC_MODEL_LABEL to base or ft}"
: "${POC_RESULTS_DIR:?Set POC_RESULTS_DIR}"
: "${POC_SHARD_INDEX:?Set POC_SHARD_INDEX}"
: "${POC_NUM_SHARDS:?Set POC_NUM_SHARDS}"

python "${POC_PROJECT_ROOT}/scripts/eval_libero_manifest.py" \
  --policy-path "${POC_EVAL_POLICY_PATH}" \
  --model-label "${POC_MODEL_LABEL}" \
  --manifest "${POC_PROJECT_ROOT}/protocol/eval_manifest.jsonl" \
  --output "${POC_RESULTS_DIR}/${POC_MODEL_LABEL}-shard-${POC_SHARD_INDEX}.jsonl" \
  --video-dir "${POC_RESULTS_DIR}/media/paired" \
  --shard-index "${POC_SHARD_INDEX}" \
  --num-shards "${POC_NUM_SHARDS}" \
  --max-action-steps 512 \
  --temperature 1.6
