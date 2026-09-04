#!/usr/bin/env python3
"""Compatibility entry point for RLinf v0.3 OpenVLA-OFT LoRA exports.

The pinned converter strips ``.base_layer.`` from every checkpoint key before
loading a PEFT model. OpenVLA-OFT's PEFT model requires those names, so retain
them while still removing wrappers introduced by compilation or DDP.
"""

from __future__ import annotations

from collections.abc import Mapping


def normalize_peft_state_dict_keys(state_dict: Mapping) -> dict:
    import torch

    normalized = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        name = key
        for prefix in ("_orig_mod.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        normalized[name] = value
    return normalized


def to_json_compatible(value):
    from omegaconf import OmegaConf

    if OmegaConf.is_config(value):
        return to_json_compatible(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, dict):
        return {key: to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_json_compatible(item) for item in value)
    return value


def sanitize_transformers_config(model) -> None:
    config = model.config
    for name, value in vars(config).items():
        setattr(config, name, to_json_compatible(value))


def main() -> None:
    from rlinf.utils.ckpt_convertor.fsdp_convertor import convert_pt_to_hf

    original_save = convert_pt_to_hf._save_hf_checkpoint

    def save_with_json_compatible_config(model, model_cfg, cfg, save_path):
        sanitize_transformers_config(model)
        return original_save(model, model_cfg, cfg, save_path)

    convert_pt_to_hf._normalize_state_dict_keys = normalize_peft_state_dict_keys
    convert_pt_to_hf._save_hf_checkpoint = save_with_json_compatible_config
    convert_pt_to_hf.main()


if __name__ == "__main__":
    main()
