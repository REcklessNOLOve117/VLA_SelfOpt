from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellLauncherTests(unittest.TestCase):
    def test_fsdp_converter_uses_peft_compatibility_entrypoint(self) -> None:
        script = (ROOT / "scripts" / "export_checkpoint.sh").read_text()
        entrypoint = 'python "${POC_PROJECT_ROOT}/scripts/convert_openvla_lora_checkpoint.py"'
        self.assertEqual(script.count(entrypoint), 2)
        self.assertNotIn("fsdp_convertor/convert_pt_to_hf.py", script)

    def test_peft_compatibility_entrypoint_preserves_base_layer_names(self) -> None:
        script = (ROOT / "scripts" / "convert_openvla_lora_checkpoint.py").read_text()
        self.assertNotIn('replace(".base_layer.", ".")', script)
        self.assertIn("_normalize_state_dict_keys = normalize_peft_state_dict_keys", script)

    def test_gpu_launchers_require_verified_nvls_setting(self) -> None:
        for name in (
            "cluster_start.sh",
            "run_benchmark.sh",
            "run_full_canary.sh",
            "run_grpo_canary.sh",
            "run_train.sh",
        ):
            with self.subTest(name=name):
                script = (ROOT / "scripts" / name).read_text()
                self.assertIn('NCCL_NVLS_ENABLE:?Set NCCL_NVLS_ENABLE=0', script)
                self.assertIn('"${NCCL_NVLS_ENABLE}" != "0"', script)


if __name__ == "__main__":
    unittest.main()
