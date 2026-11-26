# Dino RL project plan

## Research question

Can a policy trained on a dinosaur endless runner generalize to new obstacle sequences and retain its survival performance when the game gets faster or the layout distribution changes? Does training across a range of conditions improve that performance for DQN and PPO?

The main experiment is a controlled comparison of **algorithm × training diversity**. DQN and PPO represent two different ways to learn from interaction, and a timing heuristic gives us a simple baseline that can win. The simulator is an original runner built for repeatable experiments. Browser transfer is an optional follow-up after the controlled evaluation.

### Hypotheses

- Training with varied speeds and layouts improves held-out survival under speed and layout shifts. Check this separately for each algorithm and report any cost on the nominal game.
- DQN and PPO may respond differently to the same distribution shift. Compare under equal interaction budgets and report how consistent the differences are across independent runs.

## Environment MVP

- Headless Python Gymnasium environment under `env/`, with a fixed simulation tick and optional lightweight rendering for debugging.
- Actions are run, jump, and duck. The original obstacle set includes ground hazards and low, middle, and high flying hazards. Parameterize acceleration, speed, jump dynamics, obstacle mix, and gaps.
- Use the same normalized 11-feature observation for all learned policies: runner height and vertical velocity, current speed, then relative distance, bottom height, width, and height for each of the next two obstacles. Fix the normalization across training and evaluation and preserve values beyond the nominal speed range. Record the feature order and the encoding for missing obstacles.
- Start with +1 per survived tick and −10 on collision. A collision terminates an episode. Reaching the time cap truncates it. Log survival ticks, distance, cleared obstacles, crash type, and the course seed independently of reward.
- A course seed fixes the obstacle stream independently of the policy. Seed the training algorithm separately. Before training, verify same-seed replay, jump and duck collision cases, observations, and episode endings. Check representative courses from every proposed layout and speed setting using a full-course lookahead solver or scripted replay so an unavoidable collision does not masquerade as a policy failure.

## Experiment design

| Training condition | DQN | PPO | Environment per episode |
| --- | --- | --- | --- |
| Nominal | Yes | Yes | Nominal speed curve and standard randomized layouts |
| Varied | Yes | Yes | Sample a speed multiplier from 0.90–1.10 and vary obstacle frequencies and gaps within the training range |

Train both conditions on multiple seeded courses. This isolates the added effect of varying the environment parameters from the basic benefit of seeing many tracks. Also evaluate a speed-aware timing heuristic that uses the same observation and can choose when to jump or duck. Use a random policy as a smoke test.

| Evaluation suite | Changes from the nominal game | What it measures |
| --- | --- | --- |
| Held-out nominal | New course seeds drawn from the nominal generator | Generalization to new tracks from the same distribution |
| Faster game | Same course layouts, speed curve multiplied by 1.15 and 1.25 | Robustness to less reaction time at speeds beyond training |
| Held-out obstacle sequences | Nominal speed. Group existing hazard types into designated sequences excluded from both training regimes. Match obstacle counts and overall gap distribution as closely as possible. | Generalization to unfamiliar combinations of familiar hazards |

Freeze the speed levels and layout rules after feasibility checks on separate pilot tracks. Keep training, development, and final course seed lists disjoint. Choose hyperparameters and checkpoints only on the development split, with a comparable tuning budget per algorithm. Evaluate frozen policies on the same final courses within each suite. Keep a combined speed-and-layout shift as a stretch test if the individual shifts are informative.

## Run and report

1. **Pilot.** Run the random and heuristic baselines, check that the stress tests are playable, and confirm that nominal performance leaves room to detect gains. Aim for a 1,500-tick episode cap and adjust it during the pilot if the cap hides meaningful differences. Time short training runs to size the full experiment.
2. **Train.** Start with five independent training seeds per algorithm and regime, plus a target of 300,000 environment ticks per run (6 million ticks in total). Confirm or revise the budget after the pilot, then use the same budget across all configurations. Hold observations and rewards fixed. Save configs, seed lists, versions, checkpoints, and evaluation logs.
3. **Evaluate.** Use 50 shared held-out course seeds per condition as the initial target. The primary metric is mean survival ticks up to the time cap. Report cap-reach rate, the change from nominal survival, and results at each shift severity. Show the per-training-seed means and their spread. The independently trained policy is the unit of replication for algorithm comparisons. Report uncertainty and avoid claiming a winner from overlapping or inconsistent results.
4. **Diagnose.** Plot survival versus speed and summarize crashes by obstacle pattern. Replay a few matched-course failures in which one policy survives and another crashes. Check whether the policy jumped late, failed to duck, or landed into a second obstacle.

## Build order and deliverables

| Milestone | Output |
| --- | --- |
| 1. Environment | Seeded simulator, vector observations, feasibility checks, random and heuristic rollouts |
| 2. Training and evaluation | Reproducible DQN/PPO commands, separated course splits, common evaluator, pilot learning curves |
| 3. Final experiment | Results for four trained configurations, robustness plots, and failure replays |
| 4. Course deliverables | Talk and written report covering the MDP, experimental controls, findings, and failure analysis |

Keep pixel observations, reward shaping, and extra algorithms outside the initial experiment. If time gets tight, preserve both training regimes, held-out nominal tracks, and the speed sweep before expanding the layout suite. The report can still answer a focused question if a baseline wins or the two algorithms are statistically hard to distinguish.

## References

- [Cobbe et al., Leveraging Procedural Generation to Benchmark Reinforcement Learning (ICML 2020)](https://proceedings.mlr.press/v119/cobbe20a.html)
- [Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice (NeurIPS 2021)](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
- [Original project proposal](IDEA.md) and [course brief](final_project.pdf)

## Team division and timeline

### Workload split (group of 3)

| Member | Primary ownership | Concrete deliverables |
| --- | --- | --- |
| **Member 1** | Environment, splits & eval harness | Merge `seeded-dino-runner`, freeze disjoint train/dev/held-out `course_seed` lists, build held-out obstacle sequences, run feasibility checks on all frozen splits, and maintain the shared 50-seed evaluation runner |
| **Member 2** | DQN pipeline & diagnostics | Implement and tune DQN (`Nominal` + `Varied`) across 5 seeds × 300k ticks, log crash types, and run matched-course failure replays (late jump, missed duck, 2nd-obstacle landing) |
| **Member 3** | PPO pipeline, plots & presentation | Implement and tune PPO (`Nominal` + `Varied`) across 5 seeds × 300k ticks, generate uncertainty/robustness plots (`mean ± std` across seeds, survival vs. speed), and coordinate slides & written report |

### Schedule (Nov 26 – Dec 22)

- **Nov 26 – Nov 29 (Milestone 1 & pilot):** Merge `seeded-dino-runner` to `main`, freeze speed/layout splits after feasibility checks, and time short DQN/PPO pilot runs to confirm the 300k-tick budget and PPO `n_steps`.
- **Nov 30 – Dec 4 (Milestone 2: tuning & shared harness):** Tune DQN and PPO hyperparameters on the dev split; finalize the shared 50-seed evaluator and diagnostic logging.
- **Dec 5 – Dec 8 (Milestone 3: full runs & sign-up):** Run all 4 configurations × 5 seeds (`20` runs total) and evaluate frozen checkpoints across all suites. **By Dec 8:** submit group info in the Google Classroom spreadsheet (1 week before the talk).
- **Dec 9 – Dec 12 (Diagnostics & plots):** Produce final tables, per-seed spread plots, survival-vs-speed curves, and matched-course failure replays. *(Optional stretch: browser transfer demo only if core results are complete.)*
- **Dec 13 – Dec 14 (Rehearsal):** Finalize the 15-minute slide deck and rehearse talk + Q&A.
- **Dec 15 (Presentation):** 15-minute talk + 5–10 min Q&A.
- **Dec 16 – Dec 22 (Written report):** Integrate Q&A feedback and full technical details; submit report within 1 week of the presentation.
