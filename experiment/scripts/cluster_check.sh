#!/usr/bin/env bash
set -euo pipefail
: "${RLINF_ROOT:?Set RLINF_ROOT to the clean pinned checkout}"
ray status
bash "${RLINF_ROOT}/ray_utils/check_ray.sh" "${POC_EXPECTED_GPUS:-16}"
