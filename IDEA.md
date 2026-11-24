# RL Final Project Idea: Automating Chrome Dino

## Overview
Train Reinforcement Learning (RL) agents to play the **Chrome Dino** endless runner game, following the empirical project track in `final_project.pdf`.

**Core Strategy:**
1. **Fast Training in Python (`gymnasium`):** Build/wrap a faithful headless Chrome Dino environment in Python so agents can train at thousands of FPS with deterministic seeds.
2. **Real-Browser Demo (Sim-to-Real Transfer):** Deploy the trained policies via a **Manifest V3 Chrome Extension** running on a locally hosted copy of the official Chromium T-Rex Runner (`http://localhost`) for zero-latency 60 FPS live evaluation and the final presentation demo.

---

## MDP Formulation

### Action Space
- `Discrete(3)`:
  - `0`: Run (no-op)
  - `1`: Jump
  - `2`: Duck (crouch on ground / fast-fall in air)
- *(Optional extension)*: Variable jump height (short tap vs. hold).

### Observation Space (Comparison Axis)
1. **Feature Vector State (~8–10 values):**
   - Dino vertical position (`y`) and vertical velocity (`vy`)
   - Current game speed
   - Next obstacle (`dx_1`, `y_1`, `width_1`, `height_1`)
   - Second obstacle (`dx_2`, `y_2`, `width_2`, `height_2`) to avoid jumping from one obstacle directly into the next
2. **Pixel State:**
   - 4 stacked `84x84` grayscale frames (Atari-style CNN input)

### Reward Function (Ablation Axis)
- **Baseline:** `+1` per frame survived, `-10` (or `0`) on collision.
- **Shaped variants:** Small penalty for jumping (`-0.05`) to discourage unnecessary hopping; optional bonus per cleared obstacle.

---

## Key Environment Mechanics to Model
- **Accelerating game speed** over time.
- **Cacti** (varying widths/heights and clusters).
- **Pterodactyls** at 3 distinct heights:
  - *Low:* Must jump over.
  - *Mid:* Can jump over or duck under.
  - *High:* Safe to run under (jumping causes collision).
- **Day/Night color inversion** (in pixel mode, starting at score 700).

---

## Experimental Roadmap (Focused Scope)
1. **Baseline:** Random agent and a simple rule-based heuristic (`jump if dx < k * speed`) to establish a performance floor.
2. **Core RL Agent(s) (1–2 algorithms, deep tuning):**
   - **Option 1 — DQN vs. PPO (Vector State):** Compare off-policy value-based (DQN/Double-DQN) vs. on-policy policy-gradient (PPO) on the 8–10 feature vector, focusing on hyperparameter sensitivity, reward shaping, and 1-obstacle vs. 2-obstacle state ablations.
   - **Option 2 — Vector-DQN vs. Pixel-CNN-DQN:** Keep the algorithm fixed (DQN) and compare feature-vector vs. 4-frame grayscale pixel inputs, focusing on sample efficiency, frame-skipping, and the score-700 day/night color-inversion failure mode.
3. **Sim-to-Real Browser Transfer:** Export the trained policy (JSON weights for MLP, ONNX for CNN) into the Chrome Extension, evaluate on the real Chromium JS Dino engine, and diagnose any sim-to-real performance gap.

---

## Deliverables & How We Will Deliver Them (`final_project.pdf`)
1. **Reproducible Codebase & Experiment Logs (`env/`, `agents/`, `train.py`, `results/`):**
   - Custom `gymnasium` Dino environment matching Chromium `runner.js` physics.
   - Multi-seed training runs with saved checkpoints, learning curves (mean ± std score), and ablation tables.
2. **Live Chrome Extension Demo (`extension/`):**
   - Manifest V3 Chrome Extension (`"world": "MAIN"`) running exported policies at 60 FPS on a locally hosted copy of Chromium's Dino game (`http://localhost`).
   - Extension popup to switch agents live (e.g., Heuristic vs. Tabular vs. DQN/PPO) + optional canvas debug overlay showing Q-values / action probabilities.
   - *(Backup: pre-recorded 60 FPS screen capture of each agent in case of presentation-day technical issues.)*
3. **15-Minute Presentation + 5–10 min Q&A (~June 22–26):**
   - Structure: MDP formulation → baseline & failure diagnostics → ablations & what worked/failed → live Chrome Extension demo.
4. **Written Report (due 1 week after presentation):**
   - Full technical methodology, hyperparameter tables, sim-to-real gap analysis, plots from `results/`, and written responses to Q&A questions.

---

## Project Path Options
Depending on time budget and what we want to emphasize:

- **Path A — Vector-State Deep Dive (Recommended Baseline) | ~20–28 hrs total:**
  - Use the compact 8–10 feature vector state (`dino_y`, `dino_vy`, `speed`, next 2 obstacles).
  - Compare **Heuristic vs. DQN (or PPO)**—or **DQN vs. PPO**—with deep ablations on reward shaping, 1-obstacle vs. 2-obstacle state, and speed-scaling generalization.
  - Export MLP weights to a tiny `.json` file for zero-dependency JS inference in the Chrome Extension.
  - *Estimated breakdown:* Env (~5–7h) + RL Training & Ablations (~6–8h) + Chrome Extension (~4–6h) + Presentation & Report (~5–7h).
  - *Pros:* Trains in minutes on CPU, deep diagnostic focus without spreading thin, effortless extension deployment.

- **Path B — Vector-DQN vs. Pixel-CNN-DQN | ~32–45 hrs total:**
  - Keep the algorithm fixed (**DQN**) and compare **Vector-DQN** vs. **Pixel-CNN-DQN** (4 stacked `84×84` grayscale frames) against the Heuristic baseline.
  - Analyze sample efficiency, frame-skip sensitivity, and the **day/night color inversion** crash at score 700 (and how normalization/augmentation fixes it).
  - Deploy CNN to the extension via `onnxruntime-web` (or local WebSocket fallback).
  - *Estimated breakdown:* Path A (~20–28h) + Pixel Env/Rendering (~4–6h) + CNN-DQN Training & Tuning (~5–7h) + ONNX/WebSocket Extension (~3–4h).
  - *Pros:* Strong visual RL story with a single-algorithm focus; *Cons:* Slower training and heavier extension setup.

- **Path C — From-Scratch DQN/PPO & Sim-to-Real Focus | ~30–42 hrs total:**
  - Implement **one** algorithm (DQN or PPO) from scratch in PyTorch (instead of Stable-Baselines3) to demonstrate algorithmic depth, benchmarked against the Heuristic baseline.
  - Emphasize exact `runner.js` physics parity in `env/` and quantitatively measure the **sim-to-real gap** between Python scores and live Chrome Extension scores.
  - *Estimated breakdown:* Physics-accurate Env (~7–9h) + From-Scratch PyTorch RL (~10–14h) + Chrome Extension & Sim-to-Real Eval (~6–9h) + Presentation & Report (~7–10h).
  - *Pros:* High algorithmic rigor for Q&A; *Cons:* More debugging time on RL implementation details.
