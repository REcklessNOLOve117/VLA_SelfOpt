#!/usr/bin/env bash
set -euo pipefail

: "${RLINF_ROOT:?Set RLINF_ROOT}"
: "${POC_PROJECT_ROOT:?Set POC_PROJECT_ROOT}"
: "${OPENVLA_BASE_PATH:?Set OPENVLA_BASE_PATH}"
: "${WAN_WM_PATH:?Set WAN_WM_PATH}"
: "${POC_RUN_DIR:?Set POC_RUN_DIR}"
export EMBODIED_PATH="${RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${RLINF_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${POC_RUN_DIR}"

CANARY_TOTAL_ENVS="${POC_CANARY_TOTAL_ENVS:-64}"
CANARY_ACTION_CHUNKS="${POC_CANARY_GRPO_ACTION_CHUNKS:-32}"
CANARY_GLOBAL_BATCH="${POC_CANARY_GLOBAL_BATCH_SIZE:-$((CANARY_TOTAL_ENVS * CANARY_ACTION_CHUNKS))}"
CANARY_MICRO_BATCH="${POC_CANARY_MICRO_BATCH_SIZE:-32}"
if (( CANARY_TOTAL_ENVS < 8 || CANARY_TOTAL_ENVS % 8 != 0 )); then
  echo "POC_CANARY_TOTAL_ENVS must be a positive multiple of group_size=8" >&2
  exit 2
fi
if (( CANARY_GLOBAL_BATCH != CANARY_TOTAL_ENVS * CANARY_ACTION_CHUNKS )); then
  echo "Canary global batch must equal total environments times action chunks" >&2
  exit 2
fi

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
  env.train.seed="${POC_CANARY_ENV_SEED:-1234}" \
  env.train.rollout_epoch=1 env.train.total_num_envs="${CANARY_TOTAL_ENVS}" \
  env.train.max_episode_steps=256 env.train.max_steps_per_rollout_epoch=256 \
  actor.micro_batch_size="${CANARY_MICRO_BATCH}" \
  actor.global_batch_size="${CANARY_GLOBAL_BATCH}" \
  2>&1 | tee "${POC_RUN_DIR}/first-update.log"

FIRST_CHECKPOINT="$(find "${POC_RUN_DIR}" -type d -name global_step_1 | head -n 1)"
test -n "${FIRST_CHECKPOINT}"
python "${POC_PROJECT_ROOT}/scripts/validate_grpo_update.py" \
  --run-dir "${POC_RUN_DIR}" \
  --min-trajectories "${CANARY_TOTAL_ENVS}" \
  --output "${POC_RUN_DIR}/first-update-validation.json"

python "${RLINF_ROOT}/examples/embodiment/train_embodied_agent.py" \
  --config-path "${POC_PROJECT_ROOT}/configs" \
  --config-name wan_libero_spatial_grpo_openvlaoft_lora32 \
  cluster.num_nodes=1 runner.max_steps=2 runner.save_interval=1 \
  runner.resume_dir="${FIRST_CHECKPOINT}" \
  env.train.seed="${POC_CANARY_ENV_SEED:-1234}" \
  env.train.rollout_epoch=1 env.train.total_num_envs="${CANARY_TOTAL_ENVS}" \
  env.train.max_episode_steps=256 env.train.max_steps_per_rollout_epoch=256 \
  actor.micro_batch_size="${CANARY_MICRO_BATCH}" \
  actor.global_batch_size="${CANARY_GLOBAL_BATCH}" \
  2>&1 | tee "${POC_RUN_DIR}/resume-update.log"

python "${POC_PROJECT_ROOT}/scripts/hash_artifacts.py" \
  "${OPENVLA_BASE_PATH}" "${WAN_WM_PATH}/model-00001.safetensors" \
  "${WAN_WM_PATH}/Wan2.2_VAE.pth" "${WAN_WM_PATH}/resnet_rm.pth" \
  --output "${POC_RUN_DIR}/assets-after.json"
