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


def main() -> None:
    from rlinf.utils.ckpt_convertor.fsdp_convertor import convert_pt_to_hf

    convert_pt_to_hf._normalize_state_dict_keys = normalize_peft_state_dict_keys
    convert_pt_to_hf.main()


if __name__ == "__main__":
    main()
