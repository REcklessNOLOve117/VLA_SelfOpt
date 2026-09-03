from __future__ import annotations

import unittest

from poclib.topology import choose_topology


def report(dual_rate: float, *, stable: bool = True) -> dict:
    return {
        "single_node": [
            {"node": "node-a", "warmup_updates": 1, "measured_updates": 5, "updates_per_hour": 2.0},
            {"node": "node-b", "warmup_updates": 1, "measured_updates": 5, "updates_per_hour": 1.8},
        ],
        "dual_node": {"node": "joint", "warmup_updates": 1, "measured_updates": 5, "updates_per_hour": dual_rate, "stable": stable, "errors": []},
    }


class TopologyTests(unittest.TestCase):
    def test_joint_passes_at_threshold(self) -> None:
        self.assertEqual(choose_topology(report(3.0))["decision"], "joint_16_gpu")

    def test_slow_joint_falls_back_to_fastest_node(self) -> None:
        decision = choose_topology(report(2.99))
        self.assertEqual(decision["decision"], "split_8_plus_8")
        self.assertEqual(decision["training_node"], "node-a")
        self.assertEqual(decision["support_node"], "node-b")

    def test_instability_forces_fallback(self) -> None:
        self.assertEqual(choose_topology(report(4.0, stable=False))["decision"], "split_8_plus_8")


if __name__ == "__main__":
    unittest.main()
