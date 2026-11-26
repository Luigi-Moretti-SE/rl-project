"""Fixed-tick, headless Gymnasium dinosaur runner with original obstacles."""

from dataclasses import dataclass
from enum import IntEnum
from itertools import islice
from math import isfinite
from numbers import Integral
from typing import Iterator

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Action(IntEnum):
    RUN = 0
    JUMP = 1
    DUCK = 2


RUNNER_WIDTH = 19
STANDING_HEIGHT = 29
DUCK_HEIGHT = 13
# name, bottom height, obstacle height, inclusive width range
OBSTACLE_TYPES = (
    ("rock", 0, 21, (17, 34)),
    ("thorn", 0, 29, (13, 23)),
    ("skimmer", 4, 12, (17, 27)),
    ("glider", 19, 14, (23, 32)),
    ("soarer", 39, 15, (20, 31)),
)
OBSERVATION_SCALES = np.array(
    [64, 16, 10, 160, 48, 32, 32, 160, 48, 32, 32], dtype=np.float32
)


@dataclass(frozen=True, slots=True)
class DinoConfig:
    initial_speed: float = 5.5
    max_speed: float = 10.0
    acceleration: float = 0.0035
    jump_velocity: int = 15
    gravity: int = 2
    fast_fall_velocity: int = 8
    gap_range: tuple[int, int] = (95, 190)
    obstacle_weights: tuple[float, ...] = (4, 2, 1, 2, 1)
    max_steps: int = 1500

    def __post_init__(self):
        if not all(isfinite(value) for value in (
            self.initial_speed, self.max_speed, self.acceleration,
        )):
            raise ValueError("Speed settings must be finite")
        if not 0 < self.initial_speed <= self.max_speed or self.acceleration < 0:
            raise ValueError("Require 0 < initial_speed <= max_speed and acceleration >= 0")
        counts = (self.jump_velocity, self.gravity, self.fast_fall_velocity, self.max_steps)
        if any(not isinstance(value, Integral) or value <= 0 for value in counts):
            raise ValueError("Jump, gravity, fast fall, and max_steps must be positive integers")
        low, high = self.gap_range
        if not all(isinstance(value, Integral) for value in (low, high)) or not 1 <= low <= high:
            raise ValueError("gap_range must contain two ordered positive integers")
        weights = self.obstacle_weights
        if (
            len(weights) != len(OBSTACLE_TYPES)
            or any(not isfinite(weight) or weight < 0 for weight in weights)
            or not 0 < sum(weights) < float("inf")
        ):
            raise ValueError("Provide a finite, nonnegative weight for each obstacle type")


@dataclass(frozen=True, slots=True)
class Obstacle:
    """World coordinates measured from the starting line and above the ground."""

    x: float
    bottom: int
    width: int
    height: int
    kind: str

    @property
    def right(self) -> float:
        return self.x + self.width


def generate_course(seed: int, config: DinoConfig) -> Iterator[Obstacle]:
    """Yield the same layout for a seed regardless of speed or agent actions."""
    rng = np.random.default_rng(seed)
    weights = np.asarray(config.obstacle_weights, dtype=np.float64)
    probabilities = weights / weights.sum()
    x = 155
    while True:
        kind, bottom, height, (min_width, max_width) = OBSTACLE_TYPES[
            rng.choice(len(OBSTACLE_TYPES), p=probabilities)
        ]
        width = int(rng.integers(min_width, max_width + 1))
        yield Obstacle(x, bottom, width, height, kind)
        x += width + int(rng.integers(config.gap_range[0], config.gap_range[1] + 1))


def advance_runner(
    height: int, velocity: int, action: Action, config: DinoConfig
) -> tuple[int, int, int]:
    """Return the feet height, velocity, and hitbox height for one tick."""
    on_ground = height == 0
    hitbox_height = STANDING_HEIGHT
    if on_ground and action == Action.JUMP:
        velocity = config.jump_velocity
    elif action == Action.DUCK:
        if on_ground:
            hitbox_height = DUCK_HEIGHT
        else:
            velocity = min(velocity, -config.fast_fall_velocity)

    height = max(0, height + velocity)
    velocity = velocity - config.gravity if height > 0 else 0
    return height, velocity, hitbox_height


def intersects_vertically(height: int, hitbox_height: int, obstacle: Obstacle) -> bool:
    return height < obstacle.bottom + obstacle.height and height + hitbox_height > obstacle.bottom


class DinoEnv(gym.Env[np.ndarray, int]):
    """Run, jump, or duck for one tick and observe up to two upcoming hazards."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: DinoConfig | None = None, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.config = DinoConfig() if config is None else config
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(11,), dtype=np.float32)
        self._needs_reset = True
        self._has_reset = False

    def reset(self, *, seed=None, options=None):
        options = {} if options is None else options
        multiplier = float(options.get("speed_multiplier", 1.0))
        if not isfinite(multiplier) or multiplier <= 0:
            raise ValueError("speed_multiplier must be positive and finite")
        speed = self.config.initial_speed * multiplier
        max_distance = self.config.max_speed * multiplier * self.config.max_steps
        if speed <= 0 or not isfinite(max_distance):
            raise ValueError("speed_multiplier puts speed or distance outside the supported range")

        super().reset(seed=seed)
        course_seed = options.get("course_seed")
        if course_seed is None:
            course_seed = seed if seed is not None else int(self.np_random.integers(2**32))
        if not isinstance(course_seed, Integral) or course_seed < 0:
            raise ValueError("course_seed must be a nonnegative integer")

        course = options.get("course")
        if course is None:
            generated = generate_course(int(course_seed), self.config)
            course = []
            for obstacle in generated:
                course.append(obstacle)
                if obstacle.x >= max_distance + RUNNER_WIDTH:
                    course.append(next(generated))
                    break
        else:
            course = tuple(course)
            previous_right = RUNNER_WIDTH
            for obstacle in course:
                if (
                    not isinstance(obstacle, Obstacle)
                    or not isfinite(obstacle.right)
                    or obstacle.x < previous_right
                    or obstacle.bottom < 0
                    or obstacle.width <= 0
                    or obstacle.height <= 0
                ):
                    raise ValueError("Course obstacles must be ordered, disjoint, and ahead of the runner")
                previous_right = obstacle.right

        self._course = tuple(course)
        self._course_seed = int(course_seed)
        self._speed_multiplier = multiplier
        self._speed = speed
        self._distance = 0.0
        self._height = 0
        self._velocity = 0
        self._hitbox_height = STANDING_HEIGHT
        self._ticks = 0
        self._next = 0
        self._crash_type = None
        self._needs_reset = False
        self._has_reset = True
        return self._observation(), self._info()

    def step(self, action: int):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        height, velocity, hitbox_height = advance_runner(
            self._height, self._velocity, Action(int(action)), self.config
        )
        distance = self._distance + self._speed
        # Sweep the horizontal hitbox over the full tick so a fast obstacle
        # crossing still counts. The vertical pose is the one after the action.
        for index in range(self._next, len(self._course)):
            obstacle = self._course[index]
            if obstacle.x >= distance + RUNNER_WIDTH:
                break
            if intersects_vertically(height, hitbox_height, obstacle):
                self._crash_type = obstacle.kind
                break
            if obstacle.right <= distance:
                self._next += 1

        self._height, self._velocity, self._hitbox_height = height, velocity, hitbox_height
        self._distance = distance
        self._ticks += 1
        self._speed = min(
            self.config.max_speed * self._speed_multiplier,
            self._speed + self.config.acceleration * self._speed_multiplier,
        )
        terminated = self._crash_type is not None
        truncated = self._ticks >= self.config.max_steps and not terminated
        self._needs_reset = terminated or truncated
        reward = -10.0 if terminated else 1.0
        return self._observation(), reward, terminated, truncated, self._info()

    def _observation(self) -> np.ndarray:
        values = [self._height, self._velocity, self._speed]
        for obstacle in self._course[self._next : self._next + 2]:
            values.extend((
                obstacle.x - self._distance - RUNNER_WIDTH,
                obstacle.bottom, obstacle.width, obstacle.height,
            ))
        values.extend([0.0] * (11 - len(values)))
        raw = np.asarray(values, dtype=np.float64)
        return (raw / (OBSERVATION_SCALES + np.abs(raw))).astype(np.float32)

    def _info(self) -> dict:
        return {
            "course_seed": self._course_seed,
            "speed_multiplier": self._speed_multiplier,
            "ticks": self._ticks,
            "survival_ticks": self._ticks - int(self._crash_type is not None),
            "distance": self._distance,
            "cleared_obstacles": self._next,
            "crash_type": self._crash_type,
        }

    def render(self):
        if self.render_mode is None:
            return None
        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before render()")

        frame = np.full((140, 400, 3), (234, 245, 241), dtype=np.uint8)
        ground = 114
        runner_x = 28
        frame[ground:] = (190, 153, 111)

        def draw(x, bottom, width, height, color):
            left, right = max(0, int(x)), min(frame.shape[1], int(x + width))
            top, base = max(0, ground - bottom - height), ground - bottom
            if left < right and top < base:
                frame[top:base, left:right] = color

        for obstacle in islice(self._course, self._next, None):
            x = obstacle.x - self._distance + runner_x
            if x >= frame.shape[1]:
                break
            color = (136, 76, 57) if obstacle.bottom == 0 else (116, 80, 155)
            draw(x, obstacle.bottom, obstacle.width, obstacle.height, color)

        runner_color = (184, 62, 70) if self._crash_type is not None else (31, 105, 94)
        draw(runner_x, self._height, RUNNER_WIDTH, self._hitbox_height, runner_color)
        return frame
