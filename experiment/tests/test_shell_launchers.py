from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShellLauncherTests(unittest.TestCase):
    def test_fsdp_converter_uses_peft_compatibility_entrypoint(self) -> None:
        script = (ROOT / "scripts" / "export_checkpoint.sh").read_text()
        entrypoint = 'python "${POC_PROJECT_ROOT}/scripts/convert_openvla_lora_checkpoint.py"'
        self.assertEqual(script.count(entrypoint), 1)
        self.assertEqual(script.count("--config-name fsdp_model_convertor"), 1)
        self.assertEqual(script.count("fsdp_convertor/config"), 1)
        self.assertIn('python "${POC_PROJECT_ROOT}/scripts/merge_openvla_adapter.py"', script)
        self.assertIn("--dtype float32", script)
        self.assertNotIn("fsdp_convertor/convert_pt_to_hf.py", script)

    def test_peft_compatibility_entrypoint_preserves_base_layer_names(self) -> None:
        script = (ROOT / "scripts" / "convert_openvla_lora_checkpoint.py").read_text()
        self.assertNotIn('replace(".base_layer.", ".")', script)
        self.assertIn("_normalize_state_dict_keys = normalize_peft_state_dict_keys", script)

    def test_export_compatibility_entrypoint_sanitizes_omegaconf_values(self) -> None:
        script = (ROOT / "scripts" / "convert_openvla_lora_checkpoint.py").read_text()
        self.assertIn("OmegaConf.to_container(value, resolve=True)", script)
        self.assertIn("sanitize_transformers_config(model)", script)

    def test_public_hf_merger_is_safe_and_offline(self) -> None:
        script = (ROOT / "scripts" / "merge_openvla_adapter.py").read_text()
        self.assertIn("merge_and_unload(safe_merge=True)", script)
        self.assertGreaterEqual(script.count("local_files_only=True"), 2)
        self.assertIn("Refusing to overwrite", script)
        self.assertIn('default="float32"', script)

    def test_equivalence_and_truth_evaluation_default_to_float32(self) -> None:
        for name in ("compare_merged_actions.py", "eval_libero_manifest.py"):
            with self.subTest(name=name):
                script = (ROOT / "scripts" / name).read_text()
                self.assertIn('default="float32"', script)

    def test_truth_evaluation_uses_stable_renderer_and_native_crash_retries(self) -> None:
        evaluator = (ROOT / "scripts" / "eval_libero_manifest.py").read_text()
        launcher = (ROOT / "scripts" / "run_eval_shard.sh").read_text()
        self.assertLess(evaluator.index('setdefault("MUJOCO_GL"'), evaluator.index("import imageio"))
        self.assertIn('setdefault("MUJOCO_GL", "osmesa")', evaluator)
        self.assertIn('"renderer_backend": RENDERER_BACKEND', evaluator)
        self.assertIn('${POC_RENDERER_BACKEND:=osmesa}', launcher)
        self.assertIn('"${return_code}" -ne 134', launcher)
        self.assertIn('"${return_code}" -ne 139', launcher)
        self.assertIn('crash_retries=$((crash_retries + 1))', launcher)
        self.assertIn("--dtype float32", launcher)

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
