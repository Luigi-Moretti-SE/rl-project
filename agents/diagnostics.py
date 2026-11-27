"""Standardized deterministic evaluator and crash-pattern diagnostics."""

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from env.dino import OBSERVATION_SCALES, Action, DinoConfig, DinoEnv, Obstacle

from .base import PolicyFn


def _decode_height_velocity(observation: np.ndarray) -> tuple[int, int]:
    """Decode integer feet height and vertical velocity from softsign features."""
    encoded = observation[:2]
    denom = np.maximum(1.0 - np.abs(encoded), 1e-6)
    values = OBSERVATION_SCALES[:2] * encoded / denom
    return int(round(float(values[0]))), int(round(float(values[1])))


def classify_crash(
    *,
    crash_type: str | None,
    height: int,
    velocity: int,
    last_action: Action,
    airborne_cleared_prev: bool,
    ticks_since_prev_landed: int | None,
) -> str | None:
    """Assign a single standardized failure tag to a collision outcome."""
    if crash_type is None:
        return None
    if crash_type == "soarer":
        return "jumped_into_soarer"
    if airborne_cleared_prev or (
        ticks_since_prev_landed is not None and ticks_since_prev_landed <= 2
    ):
        return "second_obstacle_landing"
    if crash_type == "glider" and height == 0 and last_action != Action.DUCK:
        return "missed_duck"
    if crash_type in ("rock", "thorn", "skimmer") and (
        last_action == Action.JUMP or (height > 0 and velocity >= 0)
    ):
        return "late_jump"
    return "other_collision"


def evaluate_policy(
    policy: PolicyFn,
    seeds: Sequence[int],
    *,
    speed_multiplier: float = 1.0,
    config: DinoConfig | None = None,
    courses: Sequence[Sequence[Obstacle]] | None = None,
) -> dict[str, Any]:
    """Evaluate a deterministic policy across shared course seeds or explicit courses."""
    if courses is not None and len(courses) != len(seeds):
        raise ValueError("courses length must match seeds length when provided")

    env = DinoEnv(config)
    episodes: list[dict[str, Any]] = []
    pair_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    missing_second_obstacle_episodes = 0

    try:
        for idx, course_seed in enumerate(seeds):
            options: dict[str, Any] = {
                "course_seed": int(course_seed),
                "speed_multiplier": float(speed_multiplier),
            }
            if courses is not None:
                options["course"] = courses[idx]

            obs, _ = env.reset(seed=int(course_seed), options=options)
            saw_missing_second = bool(np.all(obs[7:11] == 0.0))
            airborne_cleared_prev = False
            ticks_since_prev_landed: int | None = None
            prev_cleared = 0
            last_action = Action.RUN

            while True:
                action = Action(int(policy(obs)))
                last_action = action
                next_obs, _, terminated, truncated, info = env.step(int(action))
                if not (terminated or truncated) and np.all(next_obs[7:11] == 0.0):
                    saw_missing_second = True

                height, velocity = _decode_height_velocity(next_obs)
                on_ground = height == 0
                cleared = int(info["cleared_obstacles"])
                crash_type = info["crash_type"]

                if terminated:
                    crash_tag = classify_crash(
                        crash_type=crash_type,
                        height=height,
                        velocity=velocity,
                        last_action=last_action,
                        airborne_cleared_prev=airborne_cleared_prev,
                        ticks_since_prev_landed=ticks_since_prev_landed,
                    )
                    prev_obstacle = (
                        env._course[cleared - 1].kind if cleared > 0 else "none"
                    )
                    if crash_type is not None:
                        pair_counts[f"{prev_obstacle}->{crash_type}"] += 1
                    if crash_tag is not None:
                        tag_counts[crash_tag] += 1
                    episodes.append(
                        {
                            "course_seed": int(course_seed),
                            "speed_multiplier": float(speed_multiplier),
                            "survival_ticks": int(info["survival_ticks"]),
                            "truncated": False,
                            "cleared_obstacles": cleared,
                            "distance": float(info["distance"]),
                            "crash_type": crash_type,
                            "prev_obstacle": prev_obstacle,
                            "crash_tag": crash_tag,
                        }
                    )
                    break

                if truncated:
                    episodes.append(
                        {
                            "course_seed": int(course_seed),
                            "speed_multiplier": float(speed_multiplier),
                            "survival_ticks": int(info["survival_ticks"]),
                            "truncated": True,
                            "cleared_obstacles": cleared,
                            "distance": float(info["distance"]),
                            "crash_type": None,
                            "prev_obstacle": None,
                            "crash_tag": None,
                        }
                    )
                    break

                if cleared > prev_cleared:
                    if not on_ground:
                        airborne_cleared_prev = True
                        ticks_since_prev_landed = None
                    else:
                        airborne_cleared_prev = False
                        ticks_since_prev_landed = 0
                    prev_cleared = cleared
                elif on_ground:
                    if airborne_cleared_prev:
                        airborne_cleared_prev = False
                        ticks_since_prev_landed = 0
                    elif ticks_since_prev_landed is not None:
                        ticks_since_prev_landed += 1

                obs = next_obs

            if saw_missing_second:
                missing_second_obstacle_episodes += 1
    finally:
        env.close()

    survivals = np.asarray([ep["survival_ticks"] for ep in episodes], dtype=np.float64)
    cleared_arr = np.asarray([ep["cleared_obstacles"] for ep in episodes], dtype=np.float64)
    trunc_arr = np.asarray([1.0 if ep["truncated"] else 0.0 for ep in episodes], dtype=np.float64)

    return {
        "n_episodes": len(episodes),
        "mean_survival_ticks": float(survivals.mean()) if len(survivals) else 0.0,
        "std_survival_ticks": float(survivals.std()) if len(survivals) else 0.0,
        "cap_reach_rate": float(trunc_arr.mean()) if len(trunc_arr) else 0.0,
        "mean_cleared_obstacles": float(cleared_arr.mean()) if len(cleared_arr) else 0.0,
        "missing_second_obstacle_episodes": missing_second_obstacle_episodes,
        "crash_counts_by_pair": dict(sorted(pair_counts.items())),
        "crash_tag_counts": dict(sorted(tag_counts.items())),
        "episodes": episodes,
    }
