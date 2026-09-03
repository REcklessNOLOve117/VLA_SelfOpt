#!/usr/bin/env python3
"""Score the frozen Wan initialization dataset with the bundled binary RM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from diffsynth.models.reward_model import ResnetRewModel

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wan-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NpyTrajectoryDatasetWrapper(str(args.wan_bundle / "dataset"), enable_kir=True)
    model = ResnetRewModel(str(args.wan_bundle / "resnet_rm.pth")).eval().to(device)
    rows = [
        {
            "dataset_index": index,
            "file_name": Path(path).name,
            "kind": "kir" if "_kir" in Path(path).name else "normal",
            "probabilities": {},
        }
        for index, path in enumerate(dataset.npy_files)
    ]

    pending_frames: list[torch.Tensor] = []
    pending_keys: list[tuple[int, str]] = []

    @torch.inference_mode()
    def flush() -> None:
        if not pending_frames:
            return
        batch = torch.stack(pending_frames).mul(2.0).sub(1.0).to(device=device, dtype=torch.float32)
        probabilities = model.net(batch).reshape(-1).detach().cpu().tolist()
        for (row_index, key), probability in zip(pending_keys, probabilities):
            rows[row_index]["probabilities"][key] = float(probability)
        pending_frames.clear()
        pending_keys.clear()

    for index in range(len(dataset)):
        episode = dataset[index]
        frames = [("start", episode["start_items"][0])]
        frames.extend((f"target_{target_index}", frame) for target_index, frame in enumerate(episode["target_items"]))
        for key, frame in frames:
            pending_frames.append(frame["image"])
            pending_keys.append((index, key))
            if len(pending_frames) >= args.batch_size:
                flush()
    flush()

    for row in rows:
        values = list(row["probabilities"].values())
        row["max_probability"] = max(values)
        row["binary_positive_count"] = sum(value >= 0.5 for value in values)

    ranked = sorted(rows, key=lambda row: (-row["max_probability"], row["file_name"]))
    all_probabilities = [value for row in rows for value in row["probabilities"].values()]
    report = {
        "schema_version": 1,
        "records": len(rows),
        "normal_records": sum(row["kind"] == "normal" for row in rows),
        "kir_records": sum(row["kind"] == "kir" for row in rows),
        "records_with_binary_positive": sum(row["binary_positive_count"] > 0 for row in rows),
        "probability_min": min(all_probabilities),
        "probability_max": max(all_probabilities),
        "top_candidates": ranked[:32],
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "top_candidates"}, indent=2))


if __name__ == "__main__":
    main()
