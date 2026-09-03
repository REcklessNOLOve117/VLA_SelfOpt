#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topology-decision", type=Path, required=True)
    parser.add_argument("--asset-hashes", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--poc-commit", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--world-model-revision", required=True)
    parser.add_argument("--started-utc")
    parser.add_argument("--ended-utc")
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "poc_commit": args.poc_commit,
            "rlinf_version": "v0.3",
            "rlinf_commit": "0505431899574619da86f551bad70b71e0ea2177",
            "config_name": args.config.name,
            "config_sha256": sha256_file(args.config),
        },
        "container_image_digest": args.container_image_digest,
        "models": {
            "base_policy": {"repo": "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1", "revision": args.base_revision},
            "world_model": {"repo": "RLinf/RLinf-Wan-LIBERO-Spatial", "revision": args.world_model_revision},
            "reward_model": {"file": "resnet_rm.pth", "bundle_revision": args.world_model_revision},
        },
        "protocol": {"suite": "LIBERO-Spatial", "tasks": 10, "states_per_task": 50, "sampling_seeds": [1234, 1235, 1236], "episodes_per_model": 1500, "training_seed": 1234, "max_updates": 100, "wall_clock_hours": 72},
        "training_window": {"started_utc": args.started_utc, "ended_utc": args.ended_utc},
        "nodes": args.nodes,
        "topology": json.loads(args.topology_decision.read_text(encoding="utf-8")),
        "artifact_hashes": json.loads(args.asset_hashes.read_text(encoding="utf-8")),
        "hardware": {
            "gpu_inventory": command_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]),
            "nvidia_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        },
        "controller": {"python": platform.python_version(), "git": command_output(["git", "--version"])},
    }
    atomic_write_json(args.output, manifest)
    print(json.dumps({"run_id": args.run_id, "output": str(args.output)}))


if __name__ == "__main__":
    main()
