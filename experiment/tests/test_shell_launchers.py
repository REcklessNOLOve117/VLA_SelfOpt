from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellLauncherTests(unittest.TestCase):
    def test_fsdp_converter_runs_as_module(self) -> None:
        script = (ROOT / "scripts" / "export_checkpoint.sh").read_text()
        module = "python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf"
        self.assertEqual(script.count(module), 2)
        self.assertNotIn("fsdp_convertor/convert_pt_to_hf.py", script)

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
