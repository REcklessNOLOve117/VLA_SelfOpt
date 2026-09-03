#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.paths:
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
        if not files:
            raise SystemExit(f"No files found at {root}")
        for path in files:
            rows.append({"root": str(root.resolve()), "relative_path": str(path.relative_to(root) if root.is_dir() else path.name).replace("\\", "/"), "size": path.stat().st_size, "sha256": hash_file(path)})
    atomic_write_json(args.output, {"schema_version": 1, "files": rows})
    print(json.dumps({"files": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
