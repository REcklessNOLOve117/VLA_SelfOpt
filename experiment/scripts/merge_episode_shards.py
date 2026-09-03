#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in read_jsonl(path)]
    identities = [(str(row["model"]), str(row["episode_key"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise SystemExit("Duplicate model/episode_key found while merging shards")
    rows.sort(key=lambda row: (str(row["model"]), int(row["task_id"]), int(row["init_state_id"]), int(row["sampling_seed"])))
    digest = write_jsonl(args.output, rows)
    print(f"merged={len(rows)} sha256={digest}")


if __name__ == "__main__":
    main()
