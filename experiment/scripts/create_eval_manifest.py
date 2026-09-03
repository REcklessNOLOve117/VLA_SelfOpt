#!/usr/bin/env python3
"""Create the immutable 1,500-episode protocol and 20 paired-video registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.io import atomic_write_json, write_jsonl
from poclib.protocol import EXPECTED_EPISODES_PER_MODEL, iter_episode_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "protocol")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.output_dir / "eval_manifest.jsonl"
    registry_path = args.output_dir / "paired_videos.json"
    if not args.force and (manifest_path.exists() or registry_path.exists()):
        raise SystemExit("Protocol files already exist; pass --force to reproduce them deterministically")

    specs = [spec.to_dict() for spec in iter_episode_specs()]
    if len(specs) != EXPECTED_EPISODES_PER_MODEL:
        raise RuntimeError("Protocol cardinality invariant failed")
    digest = write_jsonl(manifest_path, specs)
    (manifest_path.with_suffix(manifest_path.suffix + ".sha256")).write_text(
        f"{digest}  {manifest_path.name}\n", encoding="ascii"
    )

    pairs = [
        {
            "episode_key": row["episode_key"],
            "task_id": row["task_id"],
            "init_state_id": row["init_state_id"],
            "sampling_seed": row["sampling_seed"],
            "base": {"success": None, "video_path": None},
            "ft": {"success": None, "video_path": None},
            "behavior_label": None,
            "annotation_note": None,
        }
        for row in specs
        if row["record_video"]
    ]
    if len(pairs) != 20:
        raise RuntimeError(f"Expected 20 video pairs, got {len(pairs)}")
    registry = {
        "schema_version": 1,
        "status": "preregistered",
        "selection_rule": "task-wise fixed keys: state 0/seed 1234 and state 25/seed 1235",
        "hero_episode_key": None,
        "pairs": pairs,
    }
    atomic_write_json(registry_path, registry)
    registry_digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    print(json.dumps({"episodes": len(specs), "video_pairs": len(pairs), "manifest_sha256": digest, "registry_sha256": registry_digest}))


if __name__ == "__main__":
    main()
