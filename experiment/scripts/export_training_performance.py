#!/usr/bin/env python3
"""Export TensorBoard training metrics and a compact health analysis.

The generated artifacts intentionally contain only run-relative paths so they
can be shared without exposing a workstation or cluster directory layout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CORE_TAGS = (
    "time/step",
    "time/generate_rollouts",
    "time/actor/run_training",
    "time/sync_weights",
    "env/episode_len",
    "env/num_trajectories",
    "env/return",
    "env/reward",
    "env/success_once",
    "rollout/rewards",
    "rollout/advantages_min",
    "rollout/advantages_mean",
    "rollout/advantages_max",
    "train/actor/approx_kl",
    "train/actor/clip_fraction",
    "train/actor/clipped_ratio",
    "train/actor/dual_cliped_ratio",
    "train/actor/entropy_loss",
    "train/actor/grad_norm",
    "train/actor/lr",
    "train/actor/policy_loss",
    "train/actor/policy_loss_abs",
    "train/actor/ratio",
    "train/actor/ratio_abs",
    "train/actor/total_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument("--label", default="canary_global_step_2")
    parser.add_argument("--run-name")
    return parser.parse_args()


def utc_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def load_scalars(run_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    event_files = sorted(run_dir.rglob("events.out.tfevents.*"))
    for phase_index, event_file in enumerate(event_files, start=1):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        relative = event_file.relative_to(run_dir).as_posix()
        for tag in sorted(accumulator.Tags().get("scalars", [])):
            for event in accumulator.Scalars(tag):
                records.append(
                    {
                        "phase": phase_index,
                        "event_file": relative,
                        "tag": tag,
                        "step": int(event.step),
                        "completed_update": int(event.step) + 1,
                        "wall_time_utc": utc_timestamp(float(event.wall_time)),
                        "wall_time_unix": float(event.wall_time),
                        "value": float(event.value),
                    }
                )
    records.sort(key=lambda row: (row["wall_time_unix"], row["tag"], row["step"]))
    return records


def latest_by_step(records: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        key = (int(record["step"]), str(record["tag"]))
        if key not in latest or record["wall_time_unix"] >= latest[key]["wall_time_unix"]:
            latest[key] = record
    result: dict[int, dict[str, float]] = defaultdict(dict)
    for (step, tag), record in latest.items():
        result[step][tag] = float(record["value"])
    return dict(sorted(result.items()))


def metric_delta(step_one: dict[str, float], step_two: dict[str, float]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for tag in sorted(set(step_one) & set(step_two)):
        before, after = step_one[tag], step_two[tag]
        absolute = after - before
        relative = None if before == 0 else absolute / abs(before)
        result[tag] = {"step_1": before, "step_2": after, "absolute": absolute, "relative": relative}
    return result


def finite_metrics(metrics: dict[str, float]) -> bool:
    return bool(metrics) and all(math.isfinite(value) for value in metrics.values())


def build_health(
    steps: dict[int, dict[str, float]],
    checkpoint_steps: list[int],
    requested_step: int,
    training_report: dict[str, Any] | None,
    merge_report: dict[str, Any] | None,
) -> dict[str, Any]:
    target = steps.get(requested_step, {})
    advantage_min = target.get("rollout/advantages_min")
    advantage_max = target.get("rollout/advantages_max")
    grad_norm = target.get("train/actor/grad_norm")
    policy_loss_abs = target.get("train/actor/policy_loss_abs")
    checks = {
        "target_checkpoint_exists": requested_step in checkpoint_steps,
        "target_metrics_finite": finite_metrics(target),
        "advantage_span_nonzero": (
            advantage_min is not None
            and advantage_max is not None
            and math.isfinite(advantage_min)
            and math.isfinite(advantage_max)
            and advantage_max > advantage_min
        ),
        "gradient_nonzero": grad_norm is not None and math.isfinite(grad_norm) and grad_norm > 0,
        "policy_signal_nonzero": (
            policy_loss_abs is not None and math.isfinite(policy_loss_abs) and policy_loss_abs > 0
        ),
        "trainable_only_lora": bool(training_report and training_report.get("trainable_only_lora") is True),
        "resume_verified": bool(training_report and training_report.get("resume_ok") is True),
        "frozen_artifacts_unchanged": bool(
            training_report and training_report.get("frozen_artifacts_unchanged") is True
        ),
        "merged_action_equivalence": bool(merge_report and merge_report.get("passed") is True),
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = (
        "phase",
        "event_file",
        "tag",
        "step",
        "completed_update",
        "wall_time_utc",
        "value",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 100:
        return f"{value:.3f}"
    if abs(value) >= 0.01:
        return f"{value:.6f}"
    return f"{value:.6g}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    updates = {int(key): value for key, value in report["updates"].items()}
    selected_tags = [tag for tag in CORE_TAGS if any(tag in metrics for metrics in updates.values())]
    lines = [
        "# Canary global_step_2 training performance",
        "",
        f"- Exported: `{report['exported_utc']}`",
        f"- Run: `{report['run_name']}`",
        f"- Checkpoints: `{report['checkpoint_steps']}`",
        f"- Health gate: **{report['health']['status']}**",
        "- Scope: diagnostic two-update Canary; not the formal FT checkpoint.",
        "",
        "## Metrics",
        "",
        "| Metric | Step 1 | Step 2 | Delta |",
        "|---|---:|---:|---:|",
    ]
    first, second = updates.get(1, {}), updates.get(2, {})
    for tag in selected_tags:
        before, after = first.get(tag), second.get(tag)
        delta = None if before is None or after is None else after - before
        lines.append(
            f"| `{tag}` | {format_value(before)} | {format_value(after)} | {format_value(delta)} |"
        )
    lines.extend(
        [
            "",
            "## Health checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
    )
    for name, passed in report["health"]["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Two updates validate optimization mechanics, not task-performance improvement.",
            "- `env/success_once` is the frozen reward model's imagined-rollout threshold signal, not LIBERO simulator success.",
            "- Step-to-step returns use newly sampled KIR/reset groups, so their difference is not a controlled learning curve.",
            "- A real capability claim requires a paired truth evaluation against the fixed Base manifest.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_scalars(args.run_dir)
    if not records:
        raise SystemExit(f"No TensorBoard scalar records found under {args.run_dir}")

    tensorboard_steps = latest_by_step(records)
    # RLinf logs a completed global update N at TensorBoard step N-1. Keep
    # both coordinate systems explicit to prevent checkpoint/metric mismatch.
    updates = {step + 1: metrics for step, metrics in tensorboard_steps.items()}
    checkpoint_steps = sorted(
        int(path.name.rsplit("_", 1)[1])
        for path in args.run_dir.rglob("global_step_*")
        if path.is_dir() and path.name.rsplit("_", 1)[1].isdigit()
    )
    checkpoint_steps = sorted(set(checkpoint_steps))
    training_report = read_json(args.run_dir / "training-report.json")
    merge_report = read_json(args.run_dir / "merge-report.json")
    acceptance = read_json(args.run_dir / "canary-acceptance.json")
    report = {
        "schema_version": 2,
        "label": args.label,
        "run_name": args.run_name or args.run_dir.name,
        "exported_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {
            "checkpoint_step": args.checkpoint_step,
            "diagnostic_only": True,
            "formal_ft": False,
        },
        "event_files": sorted({record["event_file"] for record in records}),
        "scalar_record_count": len(records),
        "checkpoint_steps": checkpoint_steps,
        "step_indexing": "TensorBoard step N-1 corresponds to completed global update N.",
        "tensorboard_steps": {str(step): values for step, values in tensorboard_steps.items()},
        "updates": {str(update): values for update, values in updates.items()},
        "update_1_to_2_delta": metric_delta(updates.get(1, {}), updates.get(2, {})),
        "health": build_health(updates, checkpoint_steps, args.checkpoint_step, training_report, merge_report),
        "artifacts": {
            "adapter_bytes": directory_size(args.run_dir / "export" / "adapter" / "lora_adapter"),
            "merged_fp32_bytes": directory_size(args.run_dir / "export" / "merged"),
            "training_report": training_report,
            "merge_report": merge_report,
            "canary_acceptance": acceptance,
        },
        "interpretation_limits": [
            "Two updates validate optimization mechanics, not task-performance improvement.",
            "env/success_once is imagined-rollout RM threshold success, not LIBERO simulator success.",
            "Each update uses newly sampled reset/KIR groups, so step-to-step return changes mix policy change and sampling variance.",
            "Task capability requires paired truth evaluation against the fixed Base manifest.",
        ],
    }
    json_path = args.output_dir / "global_step_2_training_performance.json"
    csv_path = args.output_dir / "global_step_2_tensorboard_scalars.csv"
    markdown_path = args.output_dir / "global_step_2_training_performance.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, records)
    write_markdown(markdown_path, report)
    print(
        json.dumps(
            {
                "status": report["health"]["status"],
                "scalar_records": len(records),
                "tensorboard_steps": sorted(tensorboard_steps),
                "completed_updates": sorted(updates),
                "outputs": [path.name for path in (json_path, csv_path, markdown_path)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
