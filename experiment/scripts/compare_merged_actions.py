#!/usr/bin/env python3
"""Compare PEFT adapter inference with its merged checkpoint on 100 fixed observations."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--merged-policy", type=Path, required=True)
    parser.add_argument("--wan-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Inference precision used for both adapter and merged checkpoints.",
    )
    return parser.parse_args()


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = float(np.sqrt(0.9))
    crop_width, crop_height = int(round(width * scale)), int(round(height * scale))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((224, 224), Image.Resampling.LANCZOS)


def load_samples(dataset: Path, count: int) -> list[tuple[np.ndarray, str]]:
    files = sorted(dataset.glob("*.npy"))
    if len(files) < count:
        raise RuntimeError(f"Need {count} npy trajectories, found {len(files)}")
    result = []
    for path in files[:count]:
        trajectory = np.load(path, allow_pickle=True)
        frame = trajectory[0]
        if hasattr(frame, "item") and not isinstance(frame, dict):
            frame = frame.item()
        image = np.asarray(frame["image"], dtype=np.uint8)
        instruction = str(frame.get("task", "complete the instructed manipulation task"))
        result.append((image, instruction))
    return result


def load_model(path: Path, torch_dtype: torch.dtype):
    model = AutoModelForVision2Seq.from_pretrained(
        str(path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    stats_path = path / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if hasattr(model.vision_backbone, "set_num_images_in_input"):
        model.vision_backbone.set_num_images_in_input(1)
    return model


@torch.inference_mode()
def infer(model, processor, samples: list[tuple[np.ndarray, str]], torch_dtype: torch.dtype) -> np.ndarray:
    outputs = []
    model.eval().to("cuda")
    for index, (array, instruction) in enumerate(samples):
        random.seed(9000 + index)
        np.random.seed(9000 + index)
        torch.manual_seed(9000 + index)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        batch = processor(prompt, center_crop(Image.fromarray(array).convert("RGB")))
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat((input_ids, torch.full((1, 1), 29871, device="cuda")), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device="cuda")), dim=1)
        actions, _ = model.generate_action_verl(
            input_ids=input_ids,
            pixel_values=batch["pixel_values"].to("cuda", dtype=torch_dtype),
            attention_mask=attention_mask,
            padding_idx=processor.tokenizer.pad_token_id,
            do_sample=False,
            unnorm_key="libero_spatial_no_noops",
            temperature=1.0,
        )
        outputs.append(np.asarray(actions, dtype=np.float32))
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    samples = load_samples(args.wan_dataset, args.samples)
    processor = AutoProcessor.from_pretrained(str(args.base_policy), trust_remote_code=True, local_files_only=True)
    adapter_model = PeftModel.from_pretrained(
        load_model(args.base_policy, torch_dtype), str(args.adapter), is_trainable=False
    )
    adapter_actions = infer(adapter_model, processor, samples, torch_dtype)
    del adapter_model
    gc.collect()
    torch.cuda.empty_cache()
    merged_actions = infer(load_model(args.merged_policy, torch_dtype), processor, samples, torch_dtype)
    absolute_diff = np.abs(adapter_actions - merged_actions)
    max_diff = float(np.max(absolute_diff))
    report = {
        "schema_version": 1,
        "samples": args.samples,
        "dtype": args.dtype,
        "compared_action_values": int(absolute_diff.size),
        "different_action_values": int(np.count_nonzero(absolute_diff)),
        "mean_abs_action_diff": float(np.mean(absolute_diff)),
        "max_abs_action_diff": max_diff,
        "threshold": 1e-3,
        "passed": max_diff <= 1e-3,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
