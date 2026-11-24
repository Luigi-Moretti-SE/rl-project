# AGENTS.md

## Project Context
This repository is for a Reinforcement Learning course final project (see `final_project.pdf`, `IDEA.md`, and `LEARNINGS.md`).
The goal is to train RL agents on a fast, faithful Python `gymnasium` implementation of the Chrome Dino game and deploy/evaluate them in the real Chromium Dino JS engine via a Chrome Extension.

## Planned Architecture
- `env/` — Custom `gymnasium.Env` implementation of Chrome Dino (physics, obstacle spawning, vector & pixel observation modes).
- `agents/` — RL implementations (heuristic baseline, tabular Q-learning/SARSA, DQN, PPO, CNN-DQN).
- `train.py` — Training and evaluation entrypoints, logging metrics and saving/exporting checkpoints.
- `extension/` — Manifest V3 Chrome Extension (plus local Chromium Dino HTML/JS host) to run exported policies at 60 FPS in-browser.
- `results/` — Plots, training curves, and evaluation logs for the presentation and written report.

## Guidelines for Agents
1. **Experiment Reproducibility:** Always support random seeds, log hyperparameters, and save evaluation metrics (mean/std score, episode length) so results can be cited accurately in the report.
2. **Sim-to-Real Consistency:** Keep physics constants (gravity, jump velocity, speed scaling, obstacle hitboxes) cleanly parameterized in the `gymnasium` environment so they can be matched to Chrome Dino's JS engine.
3. **Clean Separation:** Keep the environment (`env/`) decoupled from specific RL algorithms (`agents/`) and the browser extension (`extension/`).
4. **No Premature Complexity:** Start with clean, readable implementations and iterate based on empirical results and failure diagnostics.
5. **Persistent Memory (`LEARNINGS.md`):** Always read `IDEA.md` and `LEARNINGS.md` for context, and document new technical findings, failure diagnoses, hyperparameter insights, and architectural decisions in `LEARNINGS.md` (or dedicated `.md` notes) so context is never lost across sessions.
