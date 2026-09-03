#!/usr/bin/env python3
"""Run one command under a non-resetting wall-clock budget and record its outcome."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def latest_checkpoint(root: Path) -> str | None:
    candidates = []
    for path in root.rglob("global_step_*") if root.exists() else []:
        try:
            step = int(path.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        candidates.append((step, path))
    return str(max(candidates)[1].resolve()) if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=72.0)
    parser.add_argument("--grace-seconds", type=int, default=900)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("A command is required after --")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.run_dir / "budget.json"
    if state_path.exists():
        prior = json.loads(state_path.read_text(encoding="utf-8"))
        started = float(prior["started_unix"])
    else:
        started = time.time()
    deadline = started + args.hours * 3600
    state = {"schema_version": 1, "started_unix": started, "started_utc": iso(started), "deadline_unix": deadline, "deadline_utc": iso(deadline), "command": command, "status": "running"}
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    remaining = deadline - time.time()
    if remaining <= args.grace_seconds:
        raise SystemExit("Insufficient budget remains to start another process")

    process = subprocess.Popen(command)
    stop_at = deadline - args.grace_seconds
    timed_out = False
    while process.poll() is None:
        if time.time() >= stop_at:
            timed_out = True
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=args.grace_seconds)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
            break
        time.sleep(15)
    return_code = process.wait()
    ended = time.time()
    state.update({"status": "budget_reached" if timed_out else ("completed" if return_code == 0 else "failed"), "ended_unix": ended, "ended_utc": iso(ended), "return_code": return_code, "latest_complete_checkpoint": latest_checkpoint(args.run_dir)})
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if return_code != 0 and not timed_out:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
