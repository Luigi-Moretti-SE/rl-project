"""Unit tests for shared policy contracts, filtered course sampling, diagnostics, and Double DQN."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents import DEFAULT_HELD_OUT_TRANSITIONS, make_random_policy, timing_policy
from agents.course_sampler import (
    RegimeCourseSampler,
    build_filtered_course,
    is_course_playable,
)
from agents.diagnostics import classify_crash, evaluate_policy
from agents.dqn import DQNAgent, DQNConfig, load_dqn_policy
from env import Action, DinoConfig, DinoEnv, Obstacle


class ContractAndSamplerTests(unittest.TestCase):
    def test_baseline_policies_return_valid_actions(self):
        env = DinoEnv()
        obs, _ = env.reset(seed=7)
        random_policy = make_random_policy(seed=123)
        for policy in (timing_policy, random_policy):
            action = int(policy(obs))
            self.assertIn(action, (int(Action.RUN), int(Action.JUMP), int(Action.DUCK)))
        env.close()

    def test_filtered_course_excludes_transitions_and_preserves_second_obstacle(self):
        config = DinoConfig()
        excluded = set(DEFAULT_HELD_OUT_TRANSITIONS)
        for seed in (10, 42, 99):
            course = build_filtered_course(seed, config, excluded_transitions=excluded)
            for prev, nxt in zip(course, course[1:]):
                self.assertNotIn((prev.kind, nxt.kind), excluded)

        # Full 1,500-tick survival at 1.25x must never emit obs[7:11] == [0, 0, 0, 0].
        course = build_filtered_course(10, config, max_speed_multiplier=1.25)
        self.assertTrue(is_course_playable(course, config, speed_multiplier=1.25))
        env = DinoEnv(config)
        obs, _ = env.reset(
            seed=10,
            options={"course_seed": 10, "speed_multiplier": 1.25, "course": course},
        )
        self.assertFalse(np.all(obs[7:11] == 0.0))
        env.close()

    def test_varied_sampler_produces_playable_integral_configs(self):
        sampler = RegimeCourseSampler(regime="varied", sampler_seed=11, splits_path=None)
        for _ in range(5):
            spec = sampler.sample_episode()
            self.assertGreaterEqual(spec.speed_multiplier, 0.90)
            self.assertLessEqual(spec.speed_multiplier, 1.10)
            low, high = spec.config.gap_range
            self.assertIsInstance(low, int)
            self.assertIsInstance(high, int)
            self.assertTrue(is_course_playable(spec.course, spec.config, speed_multiplier=1.25))


class DiagnosticsTests(unittest.TestCase):
    def test_classify_crash_tags(self):
        self.assertIsNone(
            classify_crash(
                crash_type=None,
                height=0,
                velocity=0,
                last_action=Action.RUN,
                airborne_cleared_prev=False,
                ticks_since_prev_landed=None,
            )
        )
        self.assertEqual(
            classify_crash(
                crash_type="soarer",
                height=28,
                velocity=8,
                last_action=Action.RUN,
                airborne_cleared_prev=False,
                ticks_since_prev_landed=None,
            ),
            "jumped_into_soarer",
        )
        self.assertEqual(
            classify_crash(
                crash_type="glider",
                height=0,
                velocity=0,
                last_action=Action.RUN,
                airborne_cleared_prev=False,
                ticks_since_prev_landed=None,
            ),
            "missed_duck",
        )
        self.assertEqual(
            classify_crash(
                crash_type="rock",
                height=15,
                velocity=13,
                last_action=Action.JUMP,
                airborne_cleared_prev=False,
                ticks_since_prev_landed=None,
            ),
            "late_jump",
        )
        self.assertEqual(
            classify_crash(
                crash_type="thorn",
                height=0,
                velocity=0,
                last_action=Action.RUN,
                airborne_cleared_prev=True,
                ticks_since_prev_landed=None,
            ),
            "second_obstacle_landing",
        )

    def test_evaluate_policy_tracks_prev_obstacle_and_tags(self):
        # Single glider at x=22 -> RUN hits glider on step 1 -> prev_obstacle="none", tag="missed_duck"
        summary = evaluate_policy(
            lambda _obs: Action.RUN,
            seeds=[1],
            config=DinoConfig(max_steps=10),
            courses=[(Obstacle(22, 19, 24, 14, "glider"),)],
        )
        self.assertEqual(summary["crash_tag_counts"], {"missed_duck": 1})
        self.assertEqual(summary["episodes"][0]["prev_obstacle"], "none")

        # Tick 1 jumps over rock at x=20..25 (cleared while airborne at t=3, d=30)
        # and lands into thorn at x=155..165 on t=15 (h=15 < 25, d=150).
        summary2 = evaluate_policy(
            lambda obs: Action.JUMP if obs[0] == 0.0 else Action.RUN,
            seeds=[2],
            config=DinoConfig(initial_speed=10.0, max_speed=10.0, acceleration=0.0, max_steps=30),
            courses=[
                (
                    Obstacle(20, 0, 5, 10, "rock"),
                    Obstacle(155, 0, 10, 25, "thorn"),
                )
            ],
        )
        self.assertEqual(summary2["episodes"][0]["prev_obstacle"], "rock")
        self.assertEqual(summary2["episodes"][0]["crash_tag"], "second_obstacle_landing")


class DQNAgentTests(unittest.TestCase):
    def test_deterministic_q_values_update_and_checkpoint_roundtrip(self):
        config = DQNConfig(
            hidden_dims=(32, 32),
            buffer_size=200,
            batch_size=8,
            learning_starts=10,
            train_freq=2,
            target_update_interval=4,
        )
        a1 = DQNAgent(config, seed=5)
        a2 = DQNAgent(config, seed=5)
        sample_obs = np.linspace(-0.5, 0.5, num=11, dtype=np.float32)
        np.testing.assert_allclose(a1.q_values(sample_obs), a2.q_values(sample_obs))

        rng = np.random.default_rng(5)
        for step in range(1, 21):
            obs = rng.uniform(-0.8, 0.8, size=11).astype(np.float32)
            next_obs = rng.uniform(-0.8, 0.8, size=11).astype(np.float32)
            a1.store_transition(obs, step % 3, 1.0, next_obs, terminated=(step == 20))
            metrics = a1.maybe_update(step)
            if step >= 10 and step % 2 == 0:
                self.assertIsNotNone(metrics)
                self.assertTrue(np.isfinite(metrics["loss"]))

        expected_q = a1.q_values(sample_obs)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "dqn.pt"
            a1.save(ckpt, step=20)
            loaded = DQNAgent.load(ckpt)
            np.testing.assert_allclose(loaded.q_values(sample_obs), expected_q)
            policy = load_dqn_policy(ckpt)
            self.assertEqual(int(policy(sample_obs)), int(np.argmax(expected_q)))


if __name__ == "__main__":
    unittest.main()
