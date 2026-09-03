"""Deterministic 16-GPU versus 8+8 topology gate."""

from __future__ import annotations

from typing import Any


def _validate_run(run: dict[str, Any], label: str) -> None:
    if int(run.get("warmup_updates", 0)) < 1:
        raise ValueError(f"{label}: at least one warm-up update is required")
    if int(run.get("measured_updates", 0)) < 5:
        raise ValueError(f"{label}: at least five measured updates are required")
    if float(run.get("updates_per_hour", 0)) <= 0:
        raise ValueError(f"{label}: updates_per_hour must be positive")


def choose_topology(report: dict[str, Any], threshold: float = 1.5) -> dict[str, Any]:
    singles = report.get("single_node")
    dual = report.get("dual_node")
    if not isinstance(singles, list) or len(singles) != 2 or not isinstance(dual, dict):
        raise ValueError("Expected two single_node runs and one dual_node run")
    for index, run in enumerate(singles):
        _validate_run(run, f"single_node[{index}]")
    _validate_run(dual, "dual_node")

    best_single = max(singles, key=lambda item: (float(item["updates_per_hour"]), item.get("node", "")))
    single_rate = float(best_single["updates_per_hour"])
    dual_rate = float(dual["updates_per_hour"])
    efficiency_ratio = dual_rate / single_rate
    dual_errors = list(dual.get("errors") or [])
    dual_stable = bool(dual.get("stable", True)) and not dual_errors
    use_joint = dual_stable and efficiency_ratio >= threshold

    return {
        "schema_version": 1,
        "decision": "joint_16_gpu" if use_joint else "split_8_plus_8",
        "num_nodes": 2 if use_joint else 1,
        "training_node": None if use_joint else best_single.get("node", "node-a"),
        "support_node": None if use_joint else next(
            run.get("node") for run in singles if run.get("node") != best_single.get("node")
        ),
        "single_node_updates_per_hour": single_rate,
        "dual_node_updates_per_hour": dual_rate,
        "dual_to_single_ratio": efficiency_ratio,
        "required_ratio": threshold,
        "dual_stable": dual_stable,
        "dual_errors": dual_errors,
        "reason": "joint throughput and stability gate passed" if use_joint else "automatic fallback: throughput or stability gate failed",
    }
