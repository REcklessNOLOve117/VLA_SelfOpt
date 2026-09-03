#!/usr/bin/env python3
"""Score simulator-confirmed LIBERO success frames with the frozen Wan RM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from diffsynth.models.reward_model import ResnetRewModel
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_libero_manifest import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    infer_action_chunk,
    load_policy,
    process_action,
    set_episode_seed,
)
from libero.libero import benchmark
from poclib.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--reward-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--state-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    parser.add_argument("--max-action-steps", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.6)
    parser.add_argument("--success-window", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def score_images(rm: ResnetRewModel, images: list[np.ndarray], batch_size: int) -> tuple[list[float], list[int]]:
    probabilities: list[float] = []
    binary_rewards: list[int] = []
    for start in range(0, len(images), batch_size):
        array = np.stack(images[start : start + batch_size]).astype(np.float32)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).to("cuda") / 127.5 - 1.0
        probabilities.extend(float(value) for value in rm.net(tensor.float()).reshape(-1).detach().cpu())
        binary_rewards.extend(int(value) for value in rm.predict_rew(tensor).reshape(-1).detach().cpu())
    return probabilities, binary_rewards


def run_candidate(
    *, env, initial_state, instruction: str, model, processor, state_id: int, sampling_seed: int, args
) -> tuple[bool, list[np.ndarray], int]:
    set_episode_seed(sampling_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
    frames = [np.asarray(get_libero_image(obs), dtype=np.uint8)]
    queue: deque[np.ndarray] = deque(maxlen=8)
    for action_step in range(1, args.max_action_steps + 1):
        if not queue:
            queue.extend(infer_action_chunk(model, processor, frames[-1], instruction, args.temperature))
        obs, _, done, _ = env.step(process_action(np.asarray(queue.popleft(), dtype=np.float32)).tolist())
        frames.append(np.asarray(get_libero_image(obs), dtype=np.uint8))
        if done:
            return True, frames, action_step
    return False, frames, args.max_action_steps


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, processor = load_policy(args.policy_path)
    rm = ResnetRewModel(str(args.reward_model)).eval().to("cuda")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    attempts = []
    success = None

    for task_id in args.task_ids:
        task = suite.get_task(task_id)
        env, instruction = get_libero_env(task, "openvla", resolution=256)
        initial_states = suite.get_task_init_states(task_id)
        try:
            for state_id in args.state_ids:
                for sampling_seed in args.sampling_seeds:
                    did_succeed, frames, action_steps = run_candidate(
                        env=env,
                        initial_state=initial_states[state_id],
                        instruction=instruction,
                        model=model,
                        processor=processor,
                        state_id=state_id,
                        sampling_seed=sampling_seed,
                        args=args,
                    )
                    attempt = {
                        "task_id": task_id,
                        "init_state_id": state_id,
                        "sampling_seed": sampling_seed,
                        "success": did_succeed,
                        "action_steps": action_steps,
                    }
                    attempts.append(attempt)
                    print(json.dumps(attempt), flush=True)
                    if did_succeed:
                        success = (attempt, instruction, frames)
                        break
                if success:
                    break
        finally:
            if hasattr(env, "close"):
                env.close()
        if success:
            break

    if success is None:
        raise RuntimeError(f"No simulator-confirmed success in {len(attempts)} fixed candidates")

    success_key, instruction, frames = success
    window = frames[-args.success_window :]
    negative_prob, negative_binary = score_images(rm, [frames[0]], args.score_batch_size)
    success_prob, success_binary = score_images(rm, window, args.score_batch_size)
    initial_path = args.output_dir / "initial-negative.png"
    success_path = args.output_dir / "simulator-success.png"
    Image.fromarray(frames[0]).save(initial_path)
    Image.fromarray(frames[-1]).save(success_path)
    report = {
        "schema_version": 1,
        "status": "passed" if max(success_binary) == 1 and negative_binary[0] == 0 else "failed",
        "ground_truth_source": "LIBERO simulator done=True",
        "instruction": instruction,
        "episode": success_key,
        "attempts": attempts,
        "reward_model_sha256": sha256(args.reward_model),
        "input_transform": "uint8 HWC -> float32 CHW in [-1, 1]",
        "negative_control": {"probability": negative_prob[0], "binary_reward": negative_binary[0]},
        "success_window": {
            "frame_count": len(window),
            "probabilities": success_prob,
            "binary_rewards": success_binary,
            "max_probability": max(success_prob),
            "any_binary_positive": max(success_binary) == 1,
            "final_probability": success_prob[-1],
            "final_binary_reward": success_binary[-1],
        },
        "artifacts": {"initial_frame": initial_path.name, "success_frame": success_path.name},
    }
    atomic_write_json(args.output_dir / "rm-libero-success-report.json", report)
    print(json.dumps({"status": report["status"], "episode": success_key, **report["success_window"]}, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("RM did not separate a simulator-confirmed success from its initial negative frame")


if __name__ == "__main__":
    main()
