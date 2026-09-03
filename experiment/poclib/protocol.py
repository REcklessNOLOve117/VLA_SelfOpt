"""Frozen evaluation protocol shared by launchers, statistics, and the result site."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

NUM_TASKS = 10
NUM_STATES_PER_TASK = 50
EVAL_SEEDS = (1234, 1235, 1236)
EXPECTED_EPISODES_PER_MODEL = NUM_TASKS * NUM_STATES_PER_TASK * len(EVAL_SEEDS)


@dataclass(frozen=True)
class EpisodeSpec:
    task_id: int
    init_state_id: int
    sampling_seed: int
    episode_key: str
    record_video: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def episode_key(task_id: int, init_state_id: int, sampling_seed: int) -> str:
    return f"task-{task_id:02d}__state-{init_state_id:02d}__seed-{sampling_seed}"


def is_preregistered_video(task_id: int, init_state_id: int, sampling_seed: int) -> bool:
    return (init_state_id, sampling_seed) in {(0, 1234), (25, 1235)}


def iter_episode_specs() -> Iterator[EpisodeSpec]:
    for task_id in range(NUM_TASKS):
        for init_state_id in range(NUM_STATES_PER_TASK):
            for sampling_seed in EVAL_SEEDS:
                yield EpisodeSpec(
                    task_id=task_id,
                    init_state_id=init_state_id,
                    sampling_seed=sampling_seed,
                    episode_key=episode_key(task_id, init_state_id, sampling_seed),
                    record_video=is_preregistered_video(task_id, init_state_id, sampling_seed),
                )


def expected_keys() -> set[str]:
    return {spec.episode_key for spec in iter_episode_specs()}
