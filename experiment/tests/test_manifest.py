from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_artifact_hashes_do_not_expose_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            asset = root / "private-host-path" / "weights.bin"
            asset.parent.mkdir()
            asset.write_bytes(b"weights")
            output = root / "hashes.json"
            subprocess.check_call(
                [sys.executable, str(EXPERIMENT_ROOT / "scripts/hash_artifacts.py"), str(asset), "--output", str(output)]
            )
            row = json.loads(output.read_text(encoding="utf-8"))["files"][0]
            self.assertNotIn("root", row)
            self.assertEqual(row["asset_name"], "weights.bin")
            self.assertEqual(row["relative_path"], "weights.bin")

    def test_experiment_manifest_records_immutable_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            topology = root / "topology.json"
            hashes = root / "hashes.json"
            config = root / "config.yaml"
            output = root / "experiment_manifest.yaml"
            topology.write_text('{"selected": "single"}\n', encoding="utf-8")
            hashes.write_text('{"files": []}\n', encoding="utf-8")
            config.write_text("seed: 1234\n", encoding="utf-8")
            subprocess.check_call(
                [
                    sys.executable,
                    str(EXPERIMENT_ROOT / "scripts/record_experiment.py"),
                    "--run-id", "test-run",
                    "--topology-decision", str(topology),
                    "--asset-hashes", str(hashes),
                    "--config", str(config),
                    "--container-image-digest", "sha256:image",
                    "--poc-commit", "poc-commit",
                    "--base-revision", "base-revision",
                    "--world-model-revision", "wm-revision",
                    "--nodes", "node-a",
                    "--output", str(output),
                ]
            )
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["poc_commit"], "poc-commit")
            self.assertEqual(manifest["models"]["base_policy"]["revision"], "base-revision")
            self.assertEqual(manifest["models"]["world_model"]["revision"], "wm-revision")
            self.assertRegex(manifest["source"]["config_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
