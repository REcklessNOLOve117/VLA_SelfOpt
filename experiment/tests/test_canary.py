from __future__ import annotations

import unittest

from poclib.canary import validate_canary


def valid_reports():
    chunk = {"action_shape": [8, 8, 7], "generated_shape": [8, 3, 8, 256, 256], "finite": True, "pixel_std": 0.2, "temporal_l1": 0.1, "group_return_variance": 0.01}
    rollout = {"status": "ok", "conditions": {"condition_frame_length": 5, "normal_records": 10, "kir_records": 10, "kir_context_l1": 0.1}, "chunks": [chunk, chunk, chunk], "reward_std": 0.02, "group_return_variance": 0.01}
    training = {"update_completed": True, "checkpoint_exists": True, "trainable_only_lora": True, "lora_rank": 32, "lora_alpha": 32, "lora_dropout": 0.0, "frozen_artifacts_unchanged": True, "resume_ok": True, "finite_loss": True, "finite_grad_norm": True, "merge_max_abs_action_diff": 1e-5}
    cleanup = {"process_residue": False, "memory_delta_gib": 0.5}
    return rollout, training, cleanup


class CanaryTests(unittest.TestCase):
    def test_valid_report_passes(self) -> None:
        self.assertEqual(validate_canary(*valid_reports()), [])

    def test_constant_reward_fails(self) -> None:
        rollout, training, cleanup = valid_reports()
        rollout["reward_std"] = 0.0
        self.assertIn("reward output is constant", validate_canary(rollout, training, cleanup))


if __name__ == "__main__":
    unittest.main()
