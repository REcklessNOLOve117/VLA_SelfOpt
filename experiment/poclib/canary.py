"""Acceptance checks for the combined rollout and one-update reports."""

from __future__ import annotations

from typing import Any


def validate_canary(rollout: dict[str, Any], training: dict[str, Any], cleanup: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    conditions = rollout.get("conditions", {})
    if rollout.get("status") != "ok":
        failures.append("rollout status is not ok")
    if conditions.get("condition_frame_length") != 5:
        failures.append("condition frame length must be 5")
    if int(conditions.get("normal_records", 0)) < 1 or int(conditions.get("kir_records", 0)) < 1:
        failures.append("both normal and KIR records are required")
    if float(conditions.get("kir_context_l1", 0)) <= 1e-4:
        failures.append("KIR context frames are not distinct")
    chunks = rollout.get("chunks") or []
    if len(chunks) < 3:
        failures.append("at least three continuous chunks are required")
    for index, chunk in enumerate(chunks):
        if chunk.get("action_shape") != [8, 8, 7]:
            failures.append(f"chunk {index}: action shape must be [8, 8, 7]")
        if chunk.get("generated_shape") != [8, 3, 8, 256, 256]:
            failures.append(f"chunk {index}: generated shape must be [8, 3, 8, 256, 256]")
        if not chunk.get("finite"):
            failures.append(f"chunk {index}: non-finite tensor detected")
        if float(chunk.get("pixel_std", 0)) <= 1e-3:
            failures.append(f"chunk {index}: generated frames are visually constant")
        if float(chunk.get("temporal_l1", 0)) <= 1e-4:
            failures.append(f"chunk {index}: generated frames have no temporal change")
    if float(rollout.get("reward_std", 0)) <= 1e-4:
        failures.append("reward output is constant")
    if float(rollout.get("group_return_variance", 0)) <= 0:
        failures.append("no GRPO group has non-zero return variance")

    required_training = {
        "update_completed": True,
        "checkpoint_exists": True,
        "trainable_only_lora": True,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "frozen_artifacts_unchanged": True,
        "resume_ok": True,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            failures.append(f"training.{key} must be {expected!r}")
    if training.get("finite_loss") is not True or training.get("finite_grad_norm") is not True:
        failures.append("training loss and grad norm must be finite")
    if float(training.get("merge_max_abs_action_diff", float("inf"))) > 1e-3:
        failures.append("merged checkpoint action difference exceeds 1e-3")

    if cleanup.get("process_residue") is not False:
        failures.append("Ray or training process residue detected")
    if float(cleanup.get("memory_delta_gib", float("inf"))) > 1.0:
        failures.append("GPU memory did not return within 1 GiB of baseline")
    return failures
