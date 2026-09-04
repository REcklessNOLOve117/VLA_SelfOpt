from __future__ import annotations

import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_grpo_update.py"
SPEC = importlib.util.spec_from_file_location("validate_grpo_update", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def valid_metrics() -> dict[str, float]:
    return {
        "env/return": 3.5,
        "env/num_trajectories": 64.0,
        "rollout/rewards": 0.25,
        "rollout/advantages_min": -1.0,
        "rollout/advantages_max": 1.0,
        "train/actor/grad_norm": 0.4,
        "train/actor/policy_loss_abs": 0.1,
        "train/actor/total_loss": 0.0,
    }


def test_accepts_finite_nonzero_learning_signal() -> None:
    assert MODULE.validate_metrics(valid_metrics(), 64) == []


def test_rejects_observed_all_zero_nan_update() -> None:
    metrics = valid_metrics()
    metrics.update(
        {
            "env/return": 0.0,
            "rollout/rewards": math.nan,
            "rollout/advantages_min": math.nan,
            "rollout/advantages_max": math.nan,
            "train/actor/grad_norm": 0.0,
            "train/actor/policy_loss_abs": 0.0,
        }
    )
    failures = MODULE.validate_metrics(metrics, 64)
    assert any("non-finite" in failure for failure in failures)
    assert any("env/return" in failure for failure in failures)
    assert any("grad_norm" in failure for failure in failures)


def test_rejects_missing_metrics() -> None:
    failures = MODULE.validate_metrics({}, 64)
    assert failures == ["missing scalar tags: " + ", ".join(MODULE.REQUIRED_TAGS)]
