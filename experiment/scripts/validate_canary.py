#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.canary import validate_canary
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout_report", type=Path)
    parser.add_argument("training_report", type=Path)
    parser.add_argument("cleanup_report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("canary_acceptance.json"))
    args = parser.parse_args()
    rollout = json.loads(args.rollout_report.read_text(encoding="utf-8"))
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    cleanup = json.loads(args.cleanup_report.read_text(encoding="utf-8"))
    failures = validate_canary(rollout, training, cleanup)
    report = {"schema_version": 1, "status": "passed" if not failures else "failed", "failures": failures}
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
