"""Reproducible Double DQN training CLI for Nominal and Varied dino regimes."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from agents.base import DEFAULT_TOTAL_TICKS, EVAL_INTERVAL_TICKS
from agents.course_sampler import RegimeCourseSampler
from agents.diagnostics import evaluate_policy
from agents.dqn import DQNAgent, DQNConfig
from env.dino import DinoConfig, DinoEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("nominal", "varied"), default="nominal")
    parser.add_argument("--seed", type=int, default=0, help="training/policy seed")
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TOTAL_TICKS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL_TICKS)
    parser.add_argument("--dev-episodes", type=int, default=50)
    parser.add_argument("--splits-path", type=Path, default=Path("splits/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dqn"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/dqn"))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1_000)
    parser.add_argument("--eps-decay-steps", type=int, default=40_000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    return parser.parse_args()


def _versions_dict() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "gymnasium": gym.__version__,
        "torch": torch.__version__,
    }


def _evaluate_dev_checkpoint(
    agent: DQNAgent,
    dev_seeds: tuple[int, ...],
    nominal_config: DinoConfig,
    step: int,
    elapsed_sec: float,
) -> dict:
    summary = evaluate_policy(
        agent,
        dev_seeds,
        speed_multiplier=1.0,
        config=nominal_config,
    )
    return {
        "step": int(step),
        "mean_survival_ticks": summary["mean_survival_ticks"],
        "std_survival_ticks": summary["std_survival_ticks"],
        "cap_reach_rate": summary["cap_reach_rate"],
        "mean_cleared_obstacles": summary["mean_cleared_obstacles"],
        "missing_second_obstacle_episodes": summary["missing_second_obstacle_episodes"],
        "wall_time_sec": round(elapsed_sec, 3),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cpu" and args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    sampler = RegimeCourseSampler(
        regime=args.regime,
        sampler_seed=args.seed * 1009 + 17,
        splits_path=args.splits_path,
    )
    dev_seeds = sampler.dev_seeds[: args.dev_episodes]
    nominal_config = DinoConfig()

    dqn_config = DQNConfig(
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        eps_decay_steps=args.eps_decay_steps,
    )
    agent = DQNAgent(dqn_config, seed=args.seed, device=args.device)

    run_tag = f"dqn_{args.regime}_seed{args.seed}"
    ckpt_root = args.checkpoint_dir / run_tag
    ckpt_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints: list[dict] = []
    best_survival = -1.0
    start_time = time.perf_counter()

    initial_eval = _evaluate_dev_checkpoint(agent, dev_seeds, nominal_config, 0, 0.0)
    checkpoints.append(initial_eval)
    best_survival = initial_eval["mean_survival_ticks"]
    agent.save(ckpt_root / "best.pt", step=0, extra_metadata=initial_eval)
    print(
        f"[{run_tag}] step=0/{args.total_timesteps} "
        f"dev={initial_eval['mean_survival_ticks']:.1f}±{initial_eval['std_survival_ticks']:.1f} "
        f"cap={initial_eval['cap_reach_rate']:.2f} eps={agent.epsilon_by_step(0):.3f}",
        flush=True,
    )

    spec = sampler.sample_episode()
    env = DinoEnv(spec.config)
    obs, _ = env.reset(
        seed=spec.course_seed,
        options={
            "course_seed": spec.course_seed,
            "speed_multiplier": spec.speed_multiplier,
            "course": spec.course,
        },
    )

    recent_losses: list[float] = []
    try:
        for step in range(1, args.total_timesteps + 1):
            action = agent.act(obs, deterministic=False, step=step)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, float(reward), next_obs, terminated)

            metrics = agent.maybe_update(step)
            if metrics is not None:
                recent_losses.append(metrics["loss"])
                if len(recent_losses) > 250:
                    recent_losses.pop(0)

            if terminated or truncated:
                spec = sampler.sample_episode()
                if spec.config != env.config:
                    env.close()
                    env = DinoEnv(spec.config)
                obs, _ = env.reset(
                    seed=spec.course_seed,
                    options={
                        "course_seed": spec.course_seed,
                        "speed_multiplier": spec.speed_multiplier,
                        "course": spec.course,
                    },
                )
            else:
                obs = next_obs

            is_eval_step = (
                args.eval_interval > 0 and step % args.eval_interval == 0
            ) or step == args.total_timesteps
            if is_eval_step:
                elapsed = time.perf_counter() - start_time
                dev_metrics = _evaluate_dev_checkpoint(
                    agent, dev_seeds, nominal_config, step, elapsed
                )
                checkpoints.append(dev_metrics)
                if dev_metrics["mean_survival_ticks"] >= best_survival:
                    best_survival = dev_metrics["mean_survival_ticks"]
                    agent.save(ckpt_root / "best.pt", step=step, extra_metadata=dev_metrics)

                sps = step / max(elapsed, 1e-6)
                loss_str = f"{np.mean(recent_losses):.4f}" if recent_losses else "n/a"
                print(
                    f"[{run_tag}] step={step}/{args.total_timesteps} "
                    f"dev={dev_metrics['mean_survival_ticks']:.1f}±{dev_metrics['std_survival_ticks']:.1f} "
                    f"cap={dev_metrics['cap_reach_rate']:.2f} eps={agent.epsilon_by_step(step):.3f} "
                    f"loss={loss_str} sps={sps:.0f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        env.close()

    total_elapsed = time.perf_counter() - start_time
    final_path = agent.save(
        ckpt_root / "final.pt",
        step=args.total_timesteps,
        extra_metadata={"best_dev_survival": best_survival, "wall_time_sec": total_elapsed},
    )

    hyperparams = asdict(dqn_config)
    hyperparams["hidden_dims"] = list(hyperparams["hidden_dims"])
    log_payload = {
        "algo": "dqn",
        "regime": args.regime,
        "train_seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_interval": args.eval_interval,
        "split": sampler.split_mode,
        "dev_episodes": len(dev_seeds),
        "hyperparameters": hyperparams,
        "versions": _versions_dict(),
        "best_checkpoint": str(ckpt_root / "best.pt"),
        "final_checkpoint": str(final_path),
        "wall_time_sec": round(total_elapsed, 3),
        "throughput_sps": round(args.total_timesteps / max(total_elapsed, 1e-6), 1),
        "checkpoints": checkpoints,
    }
    log_path = args.output_dir / f"{run_tag}_dev.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log_payload, handle, indent=2)
    print(f"Saved dev log to {log_path} and checkpoints to {ckpt_root}", flush=True)


if __name__ == "__main__":
    main()
