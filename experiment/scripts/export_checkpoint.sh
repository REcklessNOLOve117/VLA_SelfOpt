#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${POC_FSDP_CHECKPOINT:?Set POC_FSDP_CHECKPOINT to the final complete .pt checkpoint}"
: "${POC_EXPORT_DIR:?Set POC_EXPORT_DIR}"
export REPO_PATH="${RLINF_ROOT}"
mkdir -p "${POC_EXPORT_DIR}/adapter" "${POC_EXPORT_DIR}/merged"

python "${RLINF_ROOT}/rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py" \
  convertor.ckpt_path="${POC_FSDP_CHECKPOINT}" \
  convertor.save_path="${POC_EXPORT_DIR}/adapter" \
  convertor.merge_lora_weighs=false \
  convertor.torch_dtype=bf16 \
  model.model_path="${OPENVLA_BASE_PATH}" \
  model.unnorm_key=libero_spatial_no_noops \
  model.is_lora=true model.lora_rank=32 model.lora_path=null

python "${RLINF_ROOT}/rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py" \
  convertor.ckpt_path="${POC_FSDP_CHECKPOINT}" \
  convertor.save_path="${POC_EXPORT_DIR}/merged" \
  convertor.merge_lora_weighs=true \
  convertor.torch_dtype=bf16 \
  model.model_path="${OPENVLA_BASE_PATH}" \
  model.unnorm_key=libero_spatial_no_noops \
  model.is_lora=true model.lora_rank=32 model.lora_path=null
