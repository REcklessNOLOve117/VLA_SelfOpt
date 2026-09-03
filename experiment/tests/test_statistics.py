from __future__ import annotations

import unittest

from poclib.protocol import iter_episode_specs
from poclib.statistics import summarize, validate_episode_rows


def synthetic_rows(improved: bool) -> list[dict]:
    rows = []
    for model in ("base", "ft"):
        for spec in iter_episode_specs():
            success = False
            if improved and model == "ft" and spec.init_state_id < 10:
                success = True
            rows.append({**spec.to_dict(), "model": model, "task_name": f"Task {spec.task_id + 1}", "success": success, "runtime_status": "ok"})
    return rows


class StatisticsTests(unittest.TestCase):
    def test_strict_pairing(self) -> None:
        rows = synthetic_rows(False)
        validate_episode_rows(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_episode_rows(rows + [rows[0]])

    def test_positive_gate(self) -> None:
        result = summarize(synthetic_rows(True), bootstrap_samples=500, bootstrap_seed=1)
        self.assertEqual(result["overall"]["performance_gate"], "passed")
        self.assertAlmostEqual(result["overall"]["delta"], 0.2)

    def test_tie_fails_gate(self) -> None:
        result = summarize(synthetic_rows(False), bootstrap_samples=200, bootstrap_seed=1)
        self.assertEqual(result["overall"]["performance_gate"], "failed")


if __name__ == "__main__":
    unittest.main()
