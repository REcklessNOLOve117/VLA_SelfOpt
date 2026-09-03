#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.io import atomic_write_json
from poclib.topology import choose_topology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("topology_decision.json"))
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()
    report = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    decision = choose_topology(report, args.threshold)
    atomic_write_json(args.output, decision)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
