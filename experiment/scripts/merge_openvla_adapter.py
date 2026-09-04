#!/usr/bin/env python3
"""Merge an exported OpenVLA-OFT adapter through its public HF loading path."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def copy_runtime_files(base_policy: Path, output: Path) -> None:
    for source in base_policy.iterdir():
        if source.is_file() and (source.suffix == ".py" or source.name in {"dataset_statistics.json", "README.md"}):
            shutil.copy2(source, output / source.name)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stale_weights = sorted(args.output.glob("model*.safetensors"))
    if stale_weights:
        raise RuntimeError(f"Refusing to overwrite {len(stale_weights)} existing merged weight files")

    model = AutoModelForVision2Seq.from_pretrained(
        str(args.base_policy),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    peft_model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor = AutoProcessor.from_pretrained(
        str(args.base_policy), trust_remote_code=True, local_files_only=True
    )
    processor.save_pretrained(args.output)
    copy_runtime_files(args.base_policy, args.output)
    print(f"Saved public-HF-path merged checkpoint to {args.output}")


if __name__ == "__main__":
    main()
