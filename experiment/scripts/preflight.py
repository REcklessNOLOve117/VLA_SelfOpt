#!/usr/bin/env python3
"""Fail closed before reserving or starting a costly GPU run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json

PINNED_COMMIT = "0505431899574619da86f551bad70b71e0ea2177"


def command_output(command: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlinf-root", type=Path, required=True)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--wan-bundle", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--node", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    commit = command_output(["git", "rev-parse", "HEAD"], args.rlinf_root)
    if commit != PINNED_COMMIT:
        failures.append(f"RLinf commit is {commit}, expected {PINNED_COMMIT}")
    required_assets = [
        args.base_policy / "config.json",
        args.base_policy / "dataset_statistics.json",
        args.wan_bundle / "model-00001.safetensors",
        args.wan_bundle / "Wan2.2_VAE.pth",
        args.wan_bundle / "resnet_rm.pth",
        args.wan_bundle / "dataset",
    ]
    for path in required_assets:
        if not path.exists():
            failures.append(f"missing asset: {path}")
    if shutil.which("nvidia-smi") is None:
        failures.append("nvidia-smi not found")
        gpu_count, compute_processes = 0, []
    else:
        gpu_rows = [row for row in command_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]).splitlines() if row.strip()]
        gpu_count = len(gpu_rows)
        process_text = command_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"])
        compute_processes = [row for row in process_text.splitlines() if row.strip() and "No running processes" not in row]
        if gpu_count != args.expected_gpus:
            failures.append(f"found {gpu_count} GPUs, expected {args.expected_gpus}")
        if compute_processes:
            failures.append("GPUs are not exclusive; active compute processes were found")
    try:
        ray_version = command_output([sys.executable, "-c", "import ray; print(ray.__version__)"])
    except Exception as exc:
        ray_version = None
        failures.append(f"Ray import failed: {exc}")

    report = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "node": args.node,
        "rlinf_commit": commit,
        "pinned_commit": PINNED_COMMIT,
        "container_image_digest": args.image_digest,
        "gpu_count": gpu_count,
        "active_compute_process_count": len(compute_processes),
        "ray_version": ray_version,
        "network_device": os.environ.get("RLINF_COMM_NET_DEVICES"),
        "failures": failures,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
