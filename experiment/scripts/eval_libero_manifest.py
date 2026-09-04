#!/usr/bin/env python3
"""Evaluate one merged OpenVLA-OFT checkpoint on an exact manifest shard.

This intentionally avoids RLinf's concurrency-count evaluation shortcut: every
result row is tied to a canonical LIBERO task/state/sampling-seed key and is
written durably after the episode completes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
OPENVLA_ROOT = Path(os.environ.get("OPENVLA_OFT_ROOT", "dependencies/openvla-oft")).resolve()
sys.path.insert(0, str(OPENVLA_ROOT))

from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

from poclib.io import read_jsonl


class PolicyEpisodeFailure(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--model-label", choices=("base", "ft"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-action-steps", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--infrastructure-retries", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Policy inference precision; Base and FT must use the same value.",
    )
    return parser.parse_args()


def set_episode_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def center_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = float(np.sqrt(0.9))
    crop_width, crop_height = int(round(width * scale)), int(round(height * scale))
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height)).resize((224, 224), Image.Resampling.LANCZOS)


def load_policy(path: Path, torch_dtype: torch.dtype):
    model = AutoModelForVision2Seq.from_pretrained(
        str(path),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().to("cuda")
    processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True, local_files_only=True)
    stats_path = path / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if "libero_spatial_no_noops" not in model.norm_stats:
        raise RuntimeError("Checkpoint does not contain unnorm_key=libero_spatial_no_noops")
    if hasattr(model.vision_backbone, "set_num_images_in_input"):
        model.vision_backbone.set_num_images_in_input(1)
    return model, processor


@torch.inference_mode()
def infer_action_chunk(model, processor, image: np.ndarray, instruction: str, temperature: float) -> np.ndarray:
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    pil_image = center_crop(Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB"))
    batch = processor(prompt, pil_image)
    input_ids = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")
    model_dtype = next(model.parameters()).dtype
    pixel_values = batch["pixel_values"].to("cuda", dtype=model_dtype)
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
        temperature=temperature,
    )
    result = np.asarray(actions, dtype=np.float32)
    if result.shape != (1, 8, 7) or not np.isfinite(result).all():
        raise PolicyEpisodeFailure(f"invalid action output: shape={result.shape}, finite={np.isfinite(result).all()}")
    return result[0]


def process_action(action: np.ndarray) -> np.ndarray:
    if action.shape != (7,) or not np.isfinite(action).all():
        raise PolicyEpisodeFailure("policy returned an invalid 7D action")
    return invert_gripper_action(normalize_gripper_action(action, binarize=True))


def save_video(path: Path, frames: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=20, codec="libx264", quality=8, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def run_episode(
    *, env, initial_state, instruction: str, model, processor, spec: dict, args: argparse.Namespace
) -> dict:
    set_episode_seed(int(spec["sampling_seed"]))
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))

    queue: deque[np.ndarray] = deque(maxlen=8)
    frames: list[np.ndarray] = []
    action_steps = 0
    success = False
    first_success_step = None
    try:
        while action_steps < args.max_action_steps:
            image = get_libero_image(obs)
            if bool(spec["record_video"]):
                frames.append(np.asarray(image, dtype=np.uint8))
            if not queue:
                queue.extend(infer_action_chunk(model, processor, image, instruction, args.temperature))
            action = process_action(np.asarray(queue.popleft(), dtype=np.float32))
            obs, _, done, _ = env.step(action.tolist())
            action_steps += 1
            if done:
                success = True
                first_success_step = action_steps
                break
        termination = "success" if success else "timeout"
        runtime_status = "ok"
        error = None
    except PolicyEpisodeFailure as exc:
        success = False
        termination = "policy_failure"
        runtime_status = "policy_failure"
        error = str(exc)

    video_path = None
    if bool(spec["record_video"]):
        video_file = args.video_dir / args.model_label / f"{spec['episode_key']}.mp4"
        save_video(video_file, frames)
        video_path = str(video_file.resolve())
    return {
        "schema_version": 1,
        "model": args.model_label,
        "inference_dtype": args.dtype,
        "task_id": int(spec["task_id"]),
        "task_name": instruction,
        "init_state_id": int(spec["init_state_id"]),
        "sampling_seed": int(spec["sampling_seed"]),
        "episode_key": spec["episode_key"],
        "success": success,
        "first_success_step": first_success_step,
        "episode_steps": action_steps,
        "termination": termination,
        "runtime_status": runtime_status,
        "error": error,
        "video_path": video_path,
    }


def append_durable(handle, row: dict) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must satisfy 0 <= index < num-shards")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(args.manifest)
    selected = [row for index, row in enumerate(manifest) if index % args.num_shards == args.shard_index]
    completed = set()
    if args.output.exists():
        completed = {str(row["episode_key"]) for row in read_jsonl(args.output)}
    selected = [row for row in selected if str(row["episode_key"]) not in completed]
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    model, processor = load_policy(args.policy_path, torch_dtype)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    current_task_id = None
    env = None
    instruction = ""
    initial_states = None
    with args.output.open("a", encoding="utf-8") as output_handle:
        for ordinal, spec in enumerate(selected, 1):
            task_id = int(spec["task_id"])
            if task_id != current_task_id:
                if env is not None and hasattr(env, "close"):
                    env.close()
                task = suite.get_task(task_id)
                env, instruction = get_libero_env(task, "openvla", resolution=256)
                initial_states = suite.get_task_init_states(task_id)
                if len(initial_states) < 50:
                    raise RuntimeError(f"task {task_id}: expected at least 50 canonical initial states")
                current_task_id = task_id
            last_error = None
            for attempt in range(args.infrastructure_retries + 1):
                try:
                    row = run_episode(
                        env=env,
                        initial_state=initial_states[int(spec["init_state_id"])],
                        instruction=instruction,
                        model=model,
                        processor=processor,
                        spec=spec,
                        args=args,
                    )
                    append_durable(output_handle, row)
                    print(json.dumps({"completed": ordinal, "remaining": len(selected) - ordinal, "episode_key": spec["episode_key"], "success": row["success"]}))
                    break
                except PolicyEpisodeFailure:
                    raise
                except Exception as exc:
                    last_error = exc
                    if attempt == args.infrastructure_retries:
                        failure_path = args.output.with_suffix(args.output.suffix + ".infrastructure_error.json")
                        failure_path.write_text(
                            json.dumps({"episode_key": spec["episode_key"], "attempts": attempt + 1, "error": repr(exc)}, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        raise RuntimeError(f"Infrastructure failure after {attempt + 1} attempts for {spec['episode_key']}") from exc
            if last_error is not None:
                print(f"Recovered infrastructure error for {spec['episode_key']}: {last_error!r}", file=sys.stderr)
    if env is not None and hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
