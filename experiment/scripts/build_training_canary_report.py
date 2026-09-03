#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def find_checkpoints(root: Path) -> set[int]:
    result = set()
    for path in root.rglob("global_step_*"):
        try:
            result.add(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return result


def load_scalars(root: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for event_file in root.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            values.setdefault(tag, []).extend(float(event.value) for event in accumulator.Scalars(tag))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--hash-before", type=Path, required=True)
    parser.add_argument("--hash-after", type=Path, required=True)
    parser.add_argument("--merge-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    before = json.loads(args.hash_before.read_text(encoding="utf-8"))
    after = json.loads(args.hash_after.read_text(encoding="utf-8"))
    merge = json.loads(args.merge_report.read_text(encoding="utf-8"))
    scalars = load_scalars(args.run_dir)
    loss_values = [value for tag, values in scalars.items() if "loss" in tag.lower() for value in values]
    grad_values = [value for tag, values in scalars.items() if "grad_norm" in tag.lower() for value in values]
    checkpoints = find_checkpoints(args.run_dir)
    report = {
        "schema_version": 1,
        "update_completed": 1 in checkpoints,
        "checkpoint_exists": bool(checkpoints),
        "resume_ok": 2 in checkpoints,
        "trainable_only_lora": audit.get("trainable_only_lora") is True,
        "lora_rank": audit.get("lora_rank"),
        "lora_alpha": audit.get("lora_alpha"),
        "lora_dropout": audit.get("lora_dropout"),
        "frozen_artifacts_unchanged": before.get("files") == after.get("files"),
        "finite_loss": bool(loss_values) and all(math.isfinite(value) for value in loss_values),
        "finite_grad_norm": bool(grad_values) and all(math.isfinite(value) for value in grad_values),
        "merge_max_abs_action_diff": float(merge.get("max_abs_action_diff", float("inf"))),
        "checkpoint_steps": sorted(checkpoints),
        "loss_tags": sorted(tag for tag in scalars if "loss" in tag.lower()),
        "grad_norm_tags": sorted(tag for tag in scalars if "grad_norm" in tag.lower()),
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
