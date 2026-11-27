# AGENTS.md

## Project Context
This repository is for a Reinforcement Learning course final project (see `final_project.pdf`, `PROJECT_PLAN.md`, and `LEARNINGS.md`).
The active goal (`PROJECT_PLAN.md`) is a controlled 2×2 generalization study (**{DQN, PPO} × {Nominal, Varied}**) on our custom seeded Python `gymnasium` dinosaur runner (`env/dino.py`), evaluating survival under held-out tracks, `1.15×`/`1.25×` speed shifts, and held-out obstacle sequences. Browser transfer via Chrome Extension is an optional follow-up after the core results are complete.

## Planned Architecture
- `env/` — Seeded Python `gymnasium.Env` (`DinoEnv`, `DinoConfig`, `OBSTACLE_TYPES`, `generate_course`).
- `splits/` — Frozen disjoint `train` / `dev` / `eval` seed manifests and held-out obstacle sequence builders.
- `eval/` — Shared evaluation harness & deterministic crash-diagnostic logger used identically by all agents.
- `agents/` — RL policies & baselines (`heuristic.py`, DQN, PPO).
- `train.py` — Reproducible multi-seed training entrypoints, dev checkpoint curves, and saved configs/models.
- `results/` — Evaluation logs, `mean ± std` uncertainty plots, survival-vs-speed curves, and failure replays.
- `extension/` — *(Optional stretch)* Manifest V3 Chrome Extension (`"world": "MAIN"`) for browser transfer after Milestone 3.

## Guidelines for Agents
1. **Experiment Reproducibility:** Always support random seeds, log hyperparameters, and save evaluation metrics (mean/std score, episode length) so results can be cited accurately in the report.
2. **Sim-to-Real Consistency:** Keep physics constants (gravity, jump velocity, speed scaling, obstacle hitboxes) cleanly parameterized in the `gymnasium` environment so they can be matched to Chrome Dino's JS engine.
3. **Clean Separation:** Keep the environment (`env/`) decoupled from specific RL algorithms (`agents/`) and the browser extension (`extension/`).
4. **No Premature Complexity:** Start with clean, readable implementations and iterate based on empirical results and failure diagnostics.
5. **Persistent Memory (`LEARNINGS.md`):** Always read `IDEA.md` and `LEARNINGS.md` for context, and document new technical findings, failure diagnoses, hyperparameter insights, and architectural decisions in `LEARNINGS.md` (or dedicated `.md` notes) so context is never lost across sessions.
