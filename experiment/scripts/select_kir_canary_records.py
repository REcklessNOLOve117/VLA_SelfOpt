#!/usr/bin/env python3
"""Select one deterministic KIR initialization record per task instruction."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(args.dataset.glob("*_kir.npy")):
        trajectory = np.load(path, allow_pickle=True)
        instruction = str(trajectory[0]["instruction"])
        grouped[instruction].append(path.name)
    if len(grouped) != 10:
        raise RuntimeError(f"Expected exactly 10 task instructions, found {len(grouped)}")
    records = [
        {"task_index": index, "instruction": instruction, "record_name": names[0], "available_records": len(names)}
        for index, (instruction, names) in enumerate(sorted(grouped.items()))
    ]
    atomic_write_json(args.output, {"schema_version": 1, "selection": "lexicographically-first-per-task", "records": records})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
