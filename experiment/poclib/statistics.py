"""Strict paired evaluation validation and pre-registered success statistics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable

from .protocol import EVAL_SEEDS, EXPECTED_EPISODES_PER_MODEL, NUM_STATES_PER_TASK, NUM_TASKS, expected_keys

VALID_MODELS = {"base", "ft"}
VALID_RUNTIME_STATUSES = {"ok", "policy_failure"}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty list")
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def validate_episode_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {"base": {}, "ft": {}}
    expected = expected_keys()
    for row_number, row in enumerate(rows, 1):
        model = str(row.get("model", "")).lower()
        if model not in VALID_MODELS:
            raise ValueError(f"row {row_number}: model must be one of {sorted(VALID_MODELS)}")
        key = str(row.get("episode_key", ""))
        if key not in expected:
            raise ValueError(f"row {row_number}: unexpected episode_key {key!r}")
        if key in indexed[model]:
            raise ValueError(f"row {row_number}: duplicate {model}/{key}")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"row {row_number}: success must be a JSON boolean")
        status = row.get("runtime_status")
        if status not in VALID_RUNTIME_STATUSES:
            raise ValueError(f"row {row_number}: runtime_status={status!r} invalid or infrastructure run incomplete")
        indexed[model][key] = row

    for model, model_rows in indexed.items():
        if len(model_rows) != EXPECTED_EPISODES_PER_MODEL:
            missing = sorted(expected - model_rows.keys())[:5]
            extra = sorted(model_rows.keys() - expected)[:5]
            raise ValueError(
                f"{model}: expected {EXPECTED_EPISODES_PER_MODEL} unique episodes, got {len(model_rows)}; "
                f"missing={missing}, extra={extra}"
            )
    return indexed


def _task_successes(indexed: dict[str, dict[str, dict[str, Any]]], model: str, task_id: int) -> list[bool]:
    rows = [
        row
        for row in indexed[model].values()
        if int(row["task_id"]) == task_id
    ]
    if len(rows) != NUM_STATES_PER_TASK * len(EVAL_SEEDS):
        raise ValueError(f"{model}/task-{task_id:02d}: expected 150 rows, got {len(rows)}")
    return [bool(row["success"]) for row in rows]


def _paired_bootstrap(
    indexed: dict[str, dict[str, dict[str, Any]]], *, samples: int, seed: int
) -> list[float]:
    if samples < 100:
        raise ValueError("At least 100 bootstrap samples are required")
    rng = random.Random(seed)
    clusters: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for task_id in range(NUM_TASKS):
        for state_id in range(NUM_STATES_PER_TASK):
            diffs = []
            for sampling_seed in EVAL_SEEDS:
                key = f"task-{task_id:02d}__state-{state_id:02d}__seed-{sampling_seed}"
                diffs.append(float(indexed["ft"][key]["success"]) - float(indexed["base"][key]["success"]))
            clusters[task_id][state_id] = diffs

    estimates: list[float] = []
    for _ in range(samples):
        task_deltas = []
        for task_id in range(NUM_TASKS):
            sampled_states = [rng.randrange(NUM_STATES_PER_TASK) for _ in range(NUM_STATES_PER_TASK)]
            values = [value for state_id in sampled_states for value in clusters[task_id][state_id]]
            task_deltas.append(sum(values) / len(values))
        estimates.append(sum(task_deltas) / NUM_TASKS)
    estimates.sort()
    return estimates


def summarize(
    rows: Iterable[dict[str, Any]], *, bootstrap_samples: int = 10_000, bootstrap_seed: int = 20260903
) -> dict[str, Any]:
    indexed = validate_episode_rows(rows)
    tasks: list[dict[str, Any]] = []
    for task_id in range(NUM_TASKS):
        base_values = _task_successes(indexed, "base", task_id)
        ft_values = _task_successes(indexed, "ft", task_id)
        base_successes = sum(base_values)
        ft_successes = sum(ft_values)
        task_name = next(
            (str(row.get("task_name")) for row in indexed["base"].values() if int(row["task_id"]) == task_id and row.get("task_name")),
            f"Task {task_id + 1}",
        )
        tasks.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "trials": len(base_values),
                "base_successes": base_successes,
                "base_sr": base_successes / len(base_values),
                "base_wilson_95": wilson_interval(base_successes, len(base_values)),
                "ft_successes": ft_successes,
                "ft_sr": ft_successes / len(ft_values),
                "ft_wilson_95": wilson_interval(ft_successes, len(ft_values)),
                "delta": ft_successes / len(ft_values) - base_successes / len(base_values),
            }
        )

    base_macro = sum(task["base_sr"] for task in tasks) / NUM_TASKS
    ft_macro = sum(task["ft_sr"] for task in tasks) / NUM_TASKS
    delta = ft_macro - base_macro
    estimates = _paired_bootstrap(indexed, samples=bootstrap_samples, seed=bootstrap_seed)
    confidence_interval = [percentile(estimates, 0.025), percentile(estimates, 0.975)]

    seed_deltas = []
    for sampling_seed in EVAL_SEEDS:
        base_values = [row["success"] for row in indexed["base"].values() if int(row["sampling_seed"]) == sampling_seed]
        ft_values = [row["success"] for row in indexed["ft"].values() if int(row["sampling_seed"]) == sampling_seed]
        seed_deltas.append(
            {
                "sampling_seed": sampling_seed,
                "base_sr": sum(base_values) / len(base_values),
                "ft_sr": sum(ft_values) / len(ft_values),
                "delta": sum(ft_values) / len(ft_values) - sum(base_values) / len(base_values),
            }
        )

    passed = confidence_interval[0] > 0.0
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol": {
            "suite": "LIBERO-Spatial",
            "tasks": NUM_TASKS,
            "states_per_task": NUM_STATES_PER_TASK,
            "sampling_seeds": list(EVAL_SEEDS),
            "episodes_per_model": EXPECTED_EPISODES_PER_MODEL,
            "sampling": {"do_sample": True, "temperature": 1.6, "top_p": 1.0, "top_k": -1},
            "action_chunk": 8,
            "max_action_steps": 512,
        },
        "overall": {
            "base_sr": base_macro,
            "ft_sr": ft_macro,
            "delta": delta,
            "paired_cluster_bootstrap_95": confidence_interval,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "performance_gate": "passed" if passed else "failed",
        },
        "tasks": tasks,
        "seed_breakdown": seed_deltas,
    }
