#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def used_memory() -> list[int]:
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    return [int(line.strip()) for line in text.splitlines() if line.strip()]


def process_residue() -> list[str]:
    text = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    markers = ("train_embodied_agent.py", "raylet", "gcs_server")
    return [line.strip() for line in text.splitlines() if any(marker in line for marker in markers) and "gpu_cleanup_check.py" not in line]


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--output", type=Path, required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--before", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    if args.mode == "snapshot":
        report = {"schema_version": 1, "used_memory_mib": used_memory()}
        atomic_write_json(args.output, report)
        print(json.dumps(report))
        return

    before = json.loads(args.before.read_text(encoding="utf-8"))["used_memory_mib"]
    deadline = time.time() + args.timeout
    while True:
        after = used_memory()
        residue = process_residue()
        deltas = [(current - baseline) / 1024 for current, baseline in zip(after, before)]
        if max(deltas, default=0.0) <= 1.0 and not residue:
            break
        if time.time() >= deadline:
            break
        time.sleep(5)
    report = {"schema_version": 1, "process_residue": bool(residue), "residue": residue, "memory_delta_gib": max(deltas, default=0.0), "used_memory_mib": after}
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
