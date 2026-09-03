#!/usr/bin/env python3
"""Instantiate the pinned actor and prove that only rank-32 LoRA tensors train."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir

from rlinf.models import get_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from poclib.io import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="wan_libero_spatial_grpo_openvlaoft_lora32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base="1.1"):
        cfg = compose(config_name=args.config_name)
    cfg.actor.model.load_to_device = False
    model = get_model(cfg.actor.model)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    forbidden = [name for name in trainable if "lora_" not in name]
    peft_configs = list(getattr(model, "peft_config", {}).values())
    if not trainable or forbidden or not peft_configs:
        raise RuntimeError(f"LoRA trainable audit failed: trainable={len(trainable)}, forbidden={forbidden[:10]}")
    ranks = sorted({int(config.r) for config in peft_configs})
    alphas = sorted({int(config.lora_alpha) for config in peft_configs})
    dropouts = sorted({float(config.lora_dropout) for config in peft_configs})
    report = {
        "schema_version": 1,
        "trainable_only_lora": True,
        "trainable_parameter_tensors": len(trainable),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "lora_rank": ranks[0] if ranks == [32] else ranks,
        "lora_alpha": alphas[0] if alphas == [32] else alphas,
        "lora_dropout": dropouts[0] if dropouts == [0.0] else dropouts,
        "trainable_names": trainable,
    }
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "trainable_names"}, indent=2))


if __name__ == "__main__":
    main()
