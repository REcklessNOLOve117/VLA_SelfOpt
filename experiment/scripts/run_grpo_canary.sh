#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${WAN_WM_PATH:?Set WAN_WM_PATH}"
: "${POC_RUN_DIR:?Set POC_RUN_DIR}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
mkdir -p "${POC_RUN_DIR}"

python "${POC_PROJECT_ROOT}/scripts/audit_lora_trainables.py" \
  --config-dir "${POC_PROJECT_ROOT}/configs" \
  --output "${POC_RUN_DIR}/lora-audit.json"

python "${POC_PROJECT_ROOT}/scripts/hash_artifacts.py" \
  "${OPENVLA_BASE_PATH}" "${WAN_WM_PATH}/model-00001.safetensors" \
  "${WAN_WM_PATH}/Wan2.2_VAE.pth" "${WAN_WM_PATH}/resnet_rm.pth" \
  --output "${POC_RUN_DIR}/assets-before.json"

python "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${POC_PROJECT_ROOT}/configs" \
  --config-name wan_libero_spatial_grpo_openvlaoft_lora32 \
  cluster.num_nodes=1 runner.max_steps=1 runner.save_interval=1 \
  env.train.rollout_epoch=1 env.train.total_num_envs=8 \
  env.train.max_episode_steps=24 env.train.max_steps_per_rollout_epoch=24 \
  actor.micro_batch_size=8 actor.global_batch_size=256 \
  2>&1 | tee "${POC_RUN_DIR}/first-update.log"

FIRST_CHECKPOINT="$(find "${POC_RUN_DIR}" -type d -name global_step_1 | head -n 1)"
test -n "${FIRST_CHECKPOINT}"
python "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${POC_PROJECT_ROOT}/configs" \
  --config-name wan_libero_spatial_grpo_openvlaoft_lora32 \
  cluster.num_nodes=1 runner.max_steps=2 runner.save_interval=1 \
  runner.resume_dir="${FIRST_CHECKPOINT}" \
  env.train.rollout_epoch=1 env.train.total_num_envs=8 \
  env.train.max_episode_steps=24 env.train.max_steps_per_rollout_epoch=24 \
  actor.micro_batch_size=8 actor.global_batch_size=256 \
  2>&1 | tee "${POC_RUN_DIR}/resume-update.log"

python "${POC_PROJECT_ROOT}/scripts/hash_artifacts.py" \
  "${OPENVLA_BASE_PATH}" "${WAN_WM_PATH}/model-00001.safetensors" \
  "${WAN_WM_PATH}/Wan2.2_VAE.pth" "${WAN_WM_PATH}/resnet_rm.pth" \
  --output "${POC_RUN_DIR}/assets-after.json"
