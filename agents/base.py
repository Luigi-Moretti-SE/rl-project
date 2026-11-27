"""Shared policy interface and benchmark constants across RL agents and eval."""

from collections.abc import Callable
from typing import Union

import numpy as np

from env.dino import Action

PolicyFn = Callable[[np.ndarray], Union[int, Action]]

DEFAULT_HELD_OUT_TRANSITIONS: tuple[tuple[str, str], ...] = (
    ("rock", "glider"),
    ("thorn", "rock"),
    ("skimmer", "thorn"),
    ("glider", "soarer"),
)

CRASH_TAGS: tuple[str, ...] = (
    "jumped_into_soarer",
    "second_obstacle_landing",
    "missed_duck",
    "late_jump",
    "other_collision",
)

EVAL_INTERVAL_TICKS: int = 25_000
DEFAULT_TOTAL_TICKS: int = 300_000


def make_random_policy(seed: int = 0) -> PolicyFn:
    """Return a seeded uniform random baseline policy over Action."""
    rng = np.random.default_rng(seed)

    def _policy(_observation: np.ndarray) -> int:
        return int(rng.integers(0, len(Action)))

    return _policy
