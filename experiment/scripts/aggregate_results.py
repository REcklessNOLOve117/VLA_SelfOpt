#!/usr/bin/env python3
"""Validate 3,000 paired episodes and build the immutable static-site bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.io import atomic_write_json, read_jsonl, write_jsonl
from poclib.statistics import summarize

BEHAVIOR_LABELS = {
    "wrong_object",
    "reach_failure",
    "grasp_failure",
    "drop",
    "placement_or_relation_failure",
    "gripper_error",
    "oscillation_or_timeout",
    "collision_or_out_of_bounds",
    "runtime_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--video-registry", type=Path, default=ROOT / "protocol" / "paired_videos.json")
    parser.add_argument("--imagined-rollout", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def publish_media(source: str | None, destination: Path, public_path: str) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        raise ValueError(f"Missing media file: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return public_path


def update_video_registry(registry: dict, indexed_rows: dict[tuple[str, str], dict], output_dir: Path) -> dict:
    updated = json.loads(json.dumps(registry))
    reversals = []
    for pair in updated["pairs"]:
        key = pair["episode_key"]
        for model in ("base", "ft"):
            row = indexed_rows[(model, key)]
            pair[model]["success"] = row["success"]
            extension = Path(row.get("video_path") or ".mp4").suffix or ".mp4"
            relative = f"media/paired/{model}/{key}{extension}"
            pair[model]["video_path"] = publish_media(
                row.get("video_path"), output_dir / relative, f"/results/{relative}"
            )
        label = pair.get("behavior_label")
        if label is not None and label not in BEHAVIOR_LABELS:
            raise ValueError(f"{key}: unsupported behavior_label={label!r}")
        if pair["base"]["success"] is False and pair["ft"]["success"] is True:
            reversals.append(key)
    updated["status"] = "complete"
    updated["hero_episode_key"] = min(reversals) if reversals else None
    return updated


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.episodes)
    summary = summarize(rows, bootstrap_samples=args.bootstrap_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "summary.json", summary)

    with (args.output_dir / "tasks.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["task_id", "task_name", "trials", "base_successes", "base_sr", "ft_successes", "ft_sr", "delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["tasks"])

    registry = json.loads(args.video_registry.read_text(encoding="utf-8"))
    indexed = {(str(row["model"]).lower(), str(row["episode_key"])): row for row in rows}
    published_registry = update_video_registry(registry, indexed, args.output_dir)
    atomic_write_json(args.output_dir / "paired_videos.json", published_registry)
    if args.imagined_rollout:
        rollout = json.loads(args.imagined_rollout.read_text(encoding="utf-8"))
        source_dir = args.imagined_rollout.parent
        for field in ("condition_frames", "generated_frames"):
            published_paths = []
            for index, source in enumerate(rollout.get(field, [])):
                source_path = (source_dir / source).resolve() if not Path(source).is_absolute() else Path(source)
                relative = f"media/imagined/{field}/{index:02d}{source_path.suffix or '.png'}"
                published_paths.append(publish_media(str(source_path), args.output_dir / relative, f"/results/{relative}"))
            rollout[field] = published_paths
        atomic_write_json(args.output_dir / "imagined_rollout.json", rollout)
    elif not (args.output_dir / "imagined_rollout.json").exists():
        atomic_write_json(
            args.output_dir / "imagined_rollout.json",
            {"schema_version": 1, "status": "awaiting_data", "episode_key": "task-00__state-00__seed-1234", "condition_frames": [], "actions": [], "generated_frames": [], "rewards": []},
        )
    public_video_lookup = {
        (model, pair["episode_key"]): pair[model]["video_path"]
        for pair in published_registry["pairs"]
        for model in ("base", "ft")
    }
    sanitized_rows = []
    for row in rows:
        copied = dict(row)
        copied["video_path"] = public_video_lookup.get((str(row["model"]).lower(), str(row["episode_key"])))
        sanitized_rows.append(copied)
    write_jsonl(args.output_dir / "episodes.jsonl", sanitized_rows)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
