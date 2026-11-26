import unittest
from dataclasses import replace
from itertools import islice

import numpy as np
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env

from agents.heuristic import timing_policy
from env import Action, DinoConfig, DinoEnv, Obstacle, generate_course
from env.dino import RUNNER_WIDTH, advance_runner, intersects_vertically


class DinoEnvTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        check_env(DinoEnv(), skip_render_check=True)

    def test_seed_replays_the_whole_episode(self):
        config = DinoConfig(max_steps=200)
        first, second = DinoEnv(config), DinoEnv(config)
        obs, info = first.reset(seed=42)
        other_obs, other_info = second.reset(seed=42)
        np.testing.assert_array_equal(obs, other_obs)
        self.assertEqual(info, other_info)
        for _ in range(config.max_steps):
            action = timing_policy(obs)
            obs, *outcome = first.step(action)
            other_obs, *other_outcome = second.step(action)
            np.testing.assert_array_equal(obs, other_obs)
            self.assertEqual(outcome, other_outcome)
            if outcome[1] or outcome[2]:
                break

    def test_speed_shift_keeps_seeded_geometry(self):
        config = DinoConfig()
        env = DinoEnv(config)
        nominal, _ = env.reset(seed=10)
        faster, _ = env.reset(seed=99, options={"course_seed": 10, "speed_multiplier": 1.25})
        self.assertGreater(faster[2], nominal[2])
        np.testing.assert_array_equal(faster[3:], nominal[3:])
        first = list(islice(generate_course(10, config), 10))
        second = list(islice(generate_course(10, replace(config, initial_speed=8.0)), 10))
        self.assertEqual(first, second)
        self.assertNotEqual(first, list(islice(generate_course(11, config), 10)))

    def test_ground_and_flying_collision_cases(self):
        env = DinoEnv(DinoConfig(max_steps=1))
        cases = (
            (Obstacle(22, 0, 20, 11, "rock"), Action.RUN, True),
            (Obstacle(22, 0, 20, 11, "rock"), Action.JUMP, False),
            (Obstacle(22, 19, 24, 14, "glider"), Action.RUN, True),
            (Obstacle(22, 19, 24, 14, "glider"), Action.DUCK, False),
            (Obstacle(22, 39, 24, 14, "soarer"), Action.RUN, False),
            (Obstacle(22, 39, 24, 14, "soarer"), Action.JUMP, True),
        )
        for obstacle, action, crashes in cases:
            with self.subTest(obstacle=obstacle.kind, action=action):
                env.reset(seed=1, options={"course": (obstacle,)})
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(terminated, crashes)
                self.assertEqual(truncated, not crashes)
                self.assertEqual(reward, -10.0 if crashes else 1.0)
                self.assertEqual(info["crash_type"], obstacle.kind if crashes else None)
                self.assertEqual(info["survival_ticks"], int(not crashes))
                with self.assertRaises(ResetNeeded):
                    env.step(Action.RUN)

    def test_fast_fall_and_swept_collision(self):
        env = DinoEnv(DinoConfig())
        env.reset(seed=1, options={"course": ()})
        jump, *_ = env.step(Action.JUMP)
        fall, *_ = env.step(Action.DUCK)
        self.assertLess(fall[0], jump[0])
        self.assertLess(fall[1], 0)

        obstacle = Obstacle(22, 0, 4, 21, "rock")
        env.reset(seed=1, options={"course": (obstacle,), "speed_multiplier": 8.0})
        _, _, terminated, _, info = env.step(Action.RUN)
        self.assertTrue(terminated)
        self.assertEqual(info["cleared_obstacles"], 0)

    def test_render_and_missing_obstacle_encoding(self):
        env = DinoEnv(render_mode="rgb_array")
        obs, _ = env.reset(seed=5, options={"course": ()})
        np.testing.assert_array_equal(obs[3:], np.zeros(8))
        frame = env.render()
        self.assertEqual(frame.shape, (140, 400, 3))
        self.assertEqual(frame.dtype, np.uint8)
        env.step(Action.JUMP)
        self.assertFalse(np.array_equal(frame, env.render()))

    def test_timing_heuristic_improves_on_running(self):
        sums = {policy: 0 for policy in (timing_policy, always_run)}
        for seed in range(8):
            for policy in sums:
                env = DinoEnv()
                obs, _ = env.reset(seed=seed)
                while True:
                    obs, _, terminated, truncated, info = env.step(policy(obs))
                    if terminated or truncated:
                        sums[policy] += info["survival_ticks"]
                        break
        self.assertGreater(sums[timing_policy], sums[always_run])

    def test_representative_speed_and_gap_shifts_are_playable(self):
        for config in (DinoConfig(), replace(DinoConfig(), gap_range=(91, 145))):
            for multiplier in (1.0, 1.15, 1.25):
                for seed in (12, 23, 34):
                    with self.subTest(gaps=config.gap_range, speed=multiplier, seed=seed):
                        self.assertTrue(course_is_playable(config, seed, multiplier))


def always_run(_):
    return Action.RUN


def course_is_playable(config, seed, speed_multiplier):
    """Enumerate reachable vertical states with full knowledge of the course."""
    course = generate_course(seed, config)
    obstacle = next(course)
    distance = 0.0
    speed = config.initial_speed * speed_multiplier
    states = {(0, 0)}
    for _ in range(config.max_steps):
        next_distance = distance + speed
        nearby = []
        while obstacle.x < next_distance + RUNNER_WIDTH:
            nearby.append(obstacle)
            if obstacle.right > next_distance:
                break
            obstacle = next(course)

        successors = set()
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
            config.max_speed * speed_multiplier, speed + config.acceleration * speed_multiplier
        )
    return True


if __name__ == "__main__":
    unittest.main()
