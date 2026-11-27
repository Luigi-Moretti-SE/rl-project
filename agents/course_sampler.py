"""Pluggable transition-filtered course builders and Nominal/Varied samplers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.dino import (
    OBSTACLE_TYPES,
    RUNNER_WIDTH,
    Action,
    DinoConfig,
    Obstacle,
    advance_runner,
    intersects_vertically,
)

from .base import DEFAULT_HELD_OUT_TRANSITIONS

DEFAULT_PILOT_TRAIN_SEEDS: tuple[int, ...] = tuple(range(10_000, 11_000))
DEFAULT_PILOT_DEV_SEEDS: tuple[int, ...] = tuple(range(2_000, 2_050))


def is_course_playable(
    course: Sequence[Obstacle],
    config: DinoConfig,
    speed_multiplier: float = 1.25,
) -> bool:
    """Return True if at least one valid vertical trajectory survives max_steps."""
    n_obstacles = len(course)
    idx = 0
    distance = 0.0
    speed = config.initial_speed * speed_multiplier
    states = {(0, 0)}

    for _ in range(config.max_steps):
        next_distance = distance + speed
        nearby: list[Obstacle] = []
        while idx < n_obstacles and course[idx].x < next_distance + RUNNER_WIDTH:
            nearby.append(course[idx])
            if course[idx].right > next_distance:
                break
            idx += 1

        successors: set[tuple[int, int]] = set()
        for height, velocity in states:
            for action in Action:
                feet, new_velocity, hitbox = advance_runner(height, velocity, action, config)
                if not any(intersects_vertically(feet, hitbox, item) for item in nearby):
                    successors.add((feet, new_velocity))
        if not successors:
            return False
        states = successors
        distance = next_distance
        speed = min(
            config.max_speed * speed_multiplier,
            speed + config.acceleration * speed_multiplier,
        )
    return True


def build_filtered_course(
    course_seed: int,
    config: DinoConfig,
    *,
    excluded_transitions: Iterable[tuple[str, str]] = DEFAULT_HELD_OUT_TRANSITIONS,
    max_speed_multiplier: float = 1.25,
) -> tuple[Obstacle, ...]:
    """Generate a finite course excluding held-out 2-obstacle transitions.

    Always appends one extra obstacle past ``max_distance + RUNNER_WIDTH`` so
    ``obs[7:11]`` stays populated throughout a full ``max_steps`` episode.
    """
    excluded = {(str(prev), str(nxt)) for prev, nxt in excluded_transitions}
    rng = np.random.default_rng(int(course_seed))
    weights = np.asarray(config.obstacle_weights, dtype=np.float64)
    probabilities = weights / weights.sum()

    max_distance = config.max_speed * max_speed_multiplier * config.max_steps
    course: list[Obstacle] = []
    prev_kind: str | None = None
    x = 155.0
    past_limit = False

    while True:
        if prev_kind is not None:
            valid_weight = sum(
                float(weights[idx])
                for idx, (kind, *_) in enumerate(OBSTACLE_TYPES)
                if (prev_kind, kind) not in excluded
            )
            if valid_weight <= 0.0:
                raise ValueError(f"No valid obstacle can follow {prev_kind!r} under exclusions")

        while True:
            idx = int(rng.choice(len(OBSTACLE_TYPES), p=probabilities))
            kind, bottom, height, (min_width, max_width) = OBSTACLE_TYPES[idx]
            if prev_kind is None or (prev_kind, kind) not in excluded:
                break

        width = int(rng.integers(min_width, max_width + 1))
        obstacle = Obstacle(x=float(x), bottom=bottom, width=width, height=height, kind=kind)
        course.append(obstacle)
        prev_kind = kind
        x += width + int(rng.integers(config.gap_range[0], config.gap_range[1] + 1))

        if past_limit:
            break
        if obstacle.x >= max_distance + RUNNER_WIDTH:
            past_limit = True

    return tuple(course)


def load_splits_manifest(path: str | Path) -> dict[str, Any]:
    """Load frozen split seeds and held-out transitions from JSON."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Single parameterized training episode specification."""

    config: DinoConfig
    course: tuple[Obstacle, ...]
    course_seed: int
    speed_multiplier: float


class RegimeCourseSampler:
    """Sample playable Nominal or Varied training courses without held-out pairs."""

    def __init__(
        self,
        regime: str = "nominal",
        *,
        sampler_seed: int = 0,
        splits_path: str | Path | None = Path("splits/manifest.json"),
        train_seeds: Sequence[int] | None = None,
        excluded_transitions: Iterable[tuple[str, str]] | None = None,
        verify_playability: bool = False,
    ) -> None:
        if regime not in ("nominal", "varied"):
            raise ValueError(f"regime must be 'nominal' or 'varied', got {regime!r}")
        self.regime = regime
        self.rng = np.random.default_rng(int(sampler_seed))
        self.verify_playability = bool(verify_playability)
        self._nominal_cache: dict[int, tuple[Obstacle, ...]] = {}

        manifest: dict[str, Any] | None = None
        if splits_path is not None and Path(splits_path).is_file():
            manifest = load_splits_manifest(splits_path)

        if train_seeds is not None:
            self.train_seeds = tuple(int(s) for s in train_seeds)
        elif manifest is not None and "train_seeds" in manifest:
            self.train_seeds = tuple(int(s) for s in manifest["train_seeds"])
        else:
            self.train_seeds = DEFAULT_PILOT_TRAIN_SEEDS

        if not self.train_seeds:
            raise ValueError("train_seeds must be non-empty")

        if manifest is not None and "dev_seeds" in manifest:
            self.dev_seeds = tuple(int(s) for s in manifest["dev_seeds"])
            self.split_mode = "dev"
        else:
            self.dev_seeds = DEFAULT_PILOT_DEV_SEEDS
            self.split_mode = "pilot_dev"

        if excluded_transitions is not None:
            self.excluded_transitions = tuple((str(a), str(b)) for a, b in excluded_transitions)
        elif manifest is not None and "held_out_transitions" in manifest:
            self.excluded_transitions = tuple(
                (str(pair[0]), str(pair[1])) for pair in manifest["held_out_transitions"]
            )
        else:
            self.excluded_transitions = DEFAULT_HELD_OUT_TRANSITIONS

        self._nominal_config = DinoConfig()

    def sample_episode(self, *, max_attempts: int = 64) -> EpisodeSpec:
        """Sample a playable training course conforming to the configured regime."""
        for _ in range(max_attempts):
            seed_index = int(self.rng.integers(0, len(self.train_seeds)))
            course_seed = self.train_seeds[seed_index]

            if self.regime == "nominal":
                config = self._nominal_config
                speed_multiplier = 1.0
                if course_seed in self._nominal_cache:
                    return EpisodeSpec(
                        config=config,
                        course=self._nominal_cache[course_seed],
                        course_seed=course_seed,
                        speed_multiplier=speed_multiplier,
                    )
            else:
                speed_multiplier = float(self.rng.uniform(0.90, 1.10))
                gap_min = int(self.rng.integers(91, 116))
                gap_max = int(self.rng.integers(150, 211))
                jittered_weights = tuple(
                    float(w * self.rng.uniform(0.75, 1.25))
                    for w in self._nominal_config.obstacle_weights
                )
                config = DinoConfig(
                    gap_range=(gap_min, gap_max),
                    obstacle_weights=jittered_weights,
                )

            course = build_filtered_course(
                course_seed,
                config,
                excluded_transitions=self.excluded_transitions,
                max_speed_multiplier=max(speed_multiplier, 1.25),
            )
            if not self.verify_playability or is_course_playable(
                course, config, speed_multiplier=1.25
            ):
                if self.regime == "nominal":
                    self._nominal_cache[course_seed] = course
                return EpisodeSpec(
                    config=config,
                    course=course,
                    course_seed=course_seed,
                    speed_multiplier=speed_multiplier,
                )

        raise RuntimeError(f"Failed to sample a playable {self.regime!r} course in {max_attempts} attempts")
