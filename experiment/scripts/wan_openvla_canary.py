#!/usr/bin/env python3
"""Real OpenVLA -> action-conditioned Wan -> ResNet RM canary on one GPU.

Run inside the pinned RLinf Wan container with this project on PYTHONPATH.  Eight
copies of one KIR state are used so the report also verifies within-GRPO-group
return variance.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from rlinf.envs.world_model.world_model_wan_env import WanEnv
from rlinf.scheduler import Worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--config-name", default="wan_libero_spatial_grpo_openvlaoft_lora32")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    return parser.parse_args()


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = float(np.sqrt(0.9))
    crop_width, crop_height = int(round(width * scale)), int(round(height * scale))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((224, 224), Image.Resampling.LANCZOS)


def tensor_to_image(frame: torch.Tensor) -> Image.Image:
    array = (((frame.float() + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(array)


def load_policy(path: Path):
    model = AutoModelForVision2Seq.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
    ).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    stats_path = path / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if hasattr(model.vision_backbone, "set_num_images_in_input"):
        model.vision_backbone.set_num_images_in_input(1)
    return model, processor


@torch.inference_mode()
def infer_one(model, processor, image: Image.Image, instruction: str, seed: int) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    batch = processor(prompt, center_crop(image))
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    pixel_values = batch["pixel_values"].to("cuda", dtype=torch.bfloat16)
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat((input_ids, torch.full((1, 1), 29871, device="cuda")), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device="cuda")), dim=1)
    actions, _ = model.generate_action_verl(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        padding_idx=processor.tokenizer.pad_token_id,
        do_sample=True,
        unnorm_key="libero_spatial_no_noops",
        temperature=1.6,
    )
    result = np.asarray(actions, dtype=np.float32)
    if result.shape != (1, 8, 7) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid OpenVLA action output: shape={result.shape}, finite={np.isfinite(result).all()}")
    return result[0]


def context_motion(current_obs: torch.Tensor) -> float:
    frames = current_obs[0, :, 0, :5].float()
    reference = frames[:, :1]
    return float(torch.mean(torch.abs(frames[:, 1:] - reference)).cpu())


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    Worker.torch_device_type = "cuda"
    started = time.perf_counter()

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(args.config_dir.resolve()), version_base="1.1"):
        full_cfg = compose(config_name=args.config_name)
    cfg = full_cfg.env.train
    cfg.total_num_envs = args.group_size
    cfg.group_size = args.group_size
    cfg.use_fixed_reset_state_ids = False
    cfg.auto_reset = False
    cfg.enable_offload = False
    cfg.video_cfg.save_video = False

    env = WanEnv(cfg, num_envs=args.group_size, seed_offset=args.seed, total_num_processes=1, worker_info=None)
    dataset_paths = [str(path) for path in env.dataset.npy_files]
    normal_indices = [index for index, path in enumerate(dataset_paths) if "_kir" not in Path(path).name]
    kir_indices = [index for index, path in enumerate(dataset_paths) if "_kir" in Path(path).name]
    if not normal_indices or not kir_indices:
        raise RuntimeError(f"KIR coverage failed: normal={len(normal_indices)}, kir={len(kir_indices)}")

    env.reset(episode_indices=np.full(args.group_size, normal_indices[0]))
    normal_context_l1 = context_motion(env.current_obs)
    env.reset(episode_indices=np.full(args.group_size, kir_indices[0]))
    kir_context_l1 = context_motion(env.current_obs)
    if kir_context_l1 <= 1e-4:
        raise RuntimeError(f"KIR context frames do not differ from the reference frame: l1={kir_context_l1}")

    condition_dir = args.output / "condition"
    generated_dir = args.output / "generated"
    condition_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)
    condition_paths = []
    for frame_index in range(5):
        path = condition_dir / f"frame-{frame_index:02d}.png"
        tensor_to_image(env.current_obs[0, :, 0, frame_index]).save(path)
        condition_paths.append(str(path.relative_to(args.output)).replace("\\", "/"))

    model, processor = load_policy(args.policy_path)
    chunk_reports = []
    all_raw_rewards: list[float] = []
    first_actions: list[list[float]] = []
    generated_paths = []
    instruction = str(env.task_descriptions[0])
    for chunk_index in range(args.chunks):
        obs = env._wrap_obs()
        actions = []
        for env_index in range(args.group_size):
            image = Image.fromarray(obs["main_images"][env_index].detach().cpu().numpy().astype(np.uint8)).convert("RGB")
            actions.append(infer_one(model, processor, image, str(obs["task_descriptions"][env_index]), args.seed + chunk_index * 100 + env_index))
        action_batch = np.stack(actions, axis=0)
        if chunk_index == 0:
            first_actions = action_batch[0].tolist()
        _, shaped_rewards, terminations, truncations, infos = env.chunk_step(action_batch)
        raw_rewards = env._infer_next_chunk_rewards().float()
        generated = env.current_obs[:, :, 0, -8:].float()
        temporal_l1 = float(torch.mean(torch.abs(generated[:, :, 1:] - generated[:, :, :-1])).cpu())
        pixel_std = float(generated.std().cpu())
        all_raw_rewards.extend(raw_rewards.detach().cpu().reshape(-1).tolist())
        if chunk_index == 0:
            for frame_index in range(8):
                path = generated_dir / f"frame-{frame_index:02d}.png"
                tensor_to_image(generated[0, :, frame_index]).save(path)
                generated_paths.append(str(path.relative_to(args.output)).replace("\\", "/"))
        chunk_reports.append(
            {
                "chunk_index": chunk_index,
                "action_shape": list(action_batch.shape),
                "generated_shape": list(generated.shape),
                "finite": bool(torch.isfinite(generated).all() and torch.isfinite(raw_rewards).all() and torch.isfinite(shaped_rewards).all()),
                "pixel_std": pixel_std,
                "temporal_l1": temporal_l1,
                "raw_rewards": raw_rewards.detach().cpu().tolist(),
                "shaped_rewards": shaped_rewards.detach().cpu().tolist(),
                "group_return_variance": float(raw_rewards.sum(dim=1).var(unbiased=False).cpu()),
                "success_once": bool(infos[0]["episode"]["success_once"].any().item()),
                "terminated": bool(terminations.any().item()),
                "truncated": bool(truncations.any().item()),
            }
        )

    reward_std = float(np.std(np.asarray(all_raw_rewards, dtype=np.float64)))
    report = {
        "schema_version": 1,
        "status": "ok",
        "seed": args.seed,
        "episode_key": "task-00__state-00__seed-1234",
        "instruction": instruction,
        "conditions": {
            "condition_frame_length": 5,
            "normal_records": len(normal_indices),
            "kir_records": len(kir_indices),
            "normal_context_l1": normal_context_l1,
            "kir_context_l1": kir_context_l1,
        },
        "chunks": chunk_reports,
        "reward_std": reward_std,
        "condition_frames": condition_paths,
        "actions": first_actions,
        "generated_frames": generated_paths,
        "rewards": chunk_reports[0]["raw_rewards"][0],
        "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "rollout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("WAN_OPENVLA_CANARY_OK " + json.dumps({"reward_std": reward_std, "output": str(args.output)}))


if __name__ == "__main__":
    main()
