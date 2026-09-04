#!/usr/bin/env bash
set -euo pipefail

: "${POC_HEAD_IP:?Set POC_HEAD_IP}"
: "${POC_NODE_RANK:?Set POC_NODE_RANK to 0 or 1}"
: "${POC_NET_DEVICE:?Set POC_NET_DEVICE to the cross-node interface}"
: "${NCCL_NVLS_ENABLE:?Set NCCL_NVLS_ENABLE=0 after the host NCCL smoke test}"
if [[ "${NCCL_NVLS_ENABLE}" != "0" ]]; then
  echo "NCCL_NVLS_ENABLE must be 0 for the verified POC runtime" >&2
  exit 2
fi

if ray status >/dev/null 2>&1; then
  echo "A Ray cluster is already active. Refusing to alter it."
  exit 3
fi

export RLINF_NODE_RANK="${POC_NODE_RANK}"
export RLINF_COMM_NET_DEVICES="${POC_NET_DEVICE}"
export NCCL_SOCKET_IFNAME="${POC_NET_DEVICE}"
export GLOO_SOCKET_IFNAME="${POC_NET_DEVICE}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

if [[ "${POC_NODE_RANK}" == "0" ]]; then
  ray start --head --port="${POC_RAY_PORT:-6379}" --node-ip-address="${POC_HEAD_IP}"
else
  ray start --address="${POC_HEAD_IP}:${POC_RAY_PORT:-6379}"
fi
