#!/usr/bin/env python3
"""Fail fast when a GRPO update did not contain a usable learning signal."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poclib.io import atomic_write_json


REQUIRED_TAGS = (
    "env/return",
    "env/num_trajectories",
    "rollout/rewards",
    "rollout/advantages_min",
    "rollout/advantages_max",
    "train/actor/grad_norm",
    "train/actor/policy_loss_abs",
    "train/actor/total_loss",
)


def load_latest_scalars(root: Path) -> dict[str, float]:
    """Load the newest value for every scalar tag across all event files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    latest: dict[str, tuple[float, int, float]] = {}
    for event_file in sorted(root.rglob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                candidate = (float(event.wall_time), int(event.step), float(event.value))
                if tag not in latest or candidate[:2] > latest[tag][:2]:
                    latest[tag] = candidate
    return {tag: value for tag, (_, _, value) in latest.items()}


def validate_metrics(metrics: dict[str, float], min_trajectories: int) -> list[str]:
    failures: list[str] = []
    missing = [tag for tag in REQUIRED_TAGS if tag not in metrics]
    if missing:
        failures.append("missing scalar tags: " + ", ".join(missing))
        return failures

    non_finite = [tag for tag in REQUIRED_TAGS if not math.isfinite(metrics[tag])]
    if non_finite:
        failures.append("non-finite scalar values: " + ", ".join(non_finite))

    if metrics["env/num_trajectories"] < min_trajectories:
        failures.append(
            f"env/num_trajectories must be >= {min_trajectories}, "
            f"got {metrics['env/num_trajectories']}"
        )
    if not math.isfinite(metrics["env/return"]) or metrics["env/return"] <= 0:
        failures.append("env/return must be finite and greater than zero")
    advantage_min = metrics["rollout/advantages_min"]
    advantage_max = metrics["rollout/advantages_max"]
    if (
        not math.isfinite(advantage_min)
        or not math.isfinite(advantage_max)
        or advantage_max <= advantage_min
    ):
        failures.append("rollout advantages must be finite with a non-zero span")
    if not math.isfinite(metrics["train/actor/grad_norm"]) or metrics[
        "train/actor/grad_norm"
    ] <= 0:
        failures.append("train/actor/grad_norm must be finite and greater than zero")
    if not math.isfinite(metrics["train/actor/policy_loss_abs"]) or metrics[
        "train/actor/policy_loss_abs"
    ] <= 0:
        failures.append(
            "train/actor/policy_loss_abs must be finite and greater than zero"
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-trajectories", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = load_latest_scalars(args.run_dir)
    failures = validate_metrics(metrics, args.min_trajectories)
    metric_snapshot = {
        tag: value if value is None or math.isfinite(value) else None
        for tag in REQUIRED_TAGS
        if (value := metrics.get(tag)) is not None
    }
    report = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "min_trajectories": args.min_trajectories,
        "metrics": metric_snapshot,
        "failures": failures,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
