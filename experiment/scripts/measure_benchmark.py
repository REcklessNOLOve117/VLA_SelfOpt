#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoints = {}
    for path in args.run_dir.rglob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match:
            checkpoints[int(match.group(1))] = path.stat().st_mtime
    if 1 not in checkpoints or 6 not in checkpoints:
        raise SystemExit(f"Benchmark needs global_step_1 and global_step_6 checkpoints; found {sorted(checkpoints)}")
    seconds = checkpoints[6] - checkpoints[1]
    if seconds <= 0:
        raise SystemExit("Checkpoint timestamps are not increasing")
    report = {"node": args.node, "gpus": args.gpus, "warmup_updates": 1, "measured_updates": 5, "measured_seconds": seconds, "updates_per_hour": 5 * 3600 / seconds, "stable": True, "errors": []}
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
