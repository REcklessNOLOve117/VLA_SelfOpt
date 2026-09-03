#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--topology-decision", type=Path, required=True)
    parser.add_argument("--asset-hashes", type=Path, required=True)
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--nodes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rlinf": {"version": "v0.3", "commit": "0505431899574619da86f551bad70b71e0ea2177"},
        "container_image_digest": args.container_image_digest,
        "models": {"base_policy": "Haozhan72/Openvla-oft-SFT-libero-spatial-traj1", "world_model": "RLinf/RLinf-Wan-LIBERO-Spatial", "reward_model": "resnet_rm.pth"},
        "protocol": {"suite": "LIBERO-Spatial", "tasks": 10, "states_per_task": 50, "sampling_seeds": [1234, 1235, 1236], "episodes_per_model": 1500, "training_seed": 1234, "max_updates": 100, "wall_clock_hours": 72},
        "nodes": args.nodes,
        "topology": json.loads(args.topology_decision.read_text(encoding="utf-8")),
        "artifact_hashes": json.loads(args.asset_hashes.read_text(encoding="utf-8")),
        "controller": {"python": platform.python_version(), "git": subprocess.check_output(["git", "--version"], text=True).strip()},
    }
    atomic_write_json(args.output, manifest)
    print(json.dumps({"run_id": args.run_id, "output": str(args.output)}))


if __name__ == "__main__":
    main()
