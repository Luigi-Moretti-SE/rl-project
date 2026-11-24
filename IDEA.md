# RL Final Project Idea: Automating Chrome Dino

## Overview
Train Reinforcement Learning (RL) agents to play the **Chrome Dino** endless runner game, following the empirical project track in `final_project.pdf`.

**Core Strategy:**
1. **Fast Training in Python (`gymnasium`):** Build/wrap a faithful headless Chrome Dino environment in Python so agents can train at thousands of FPS with deterministic seeds.
2. **Real-Browser Demo (Sim-to-Real Transfer):** Deploy the trained policies on the actual `chrome://dino` game via browser automation (Playwright/Selenium) for live evaluation and the final presentation demo.

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

## Experimental Roadmap
1. **Baselines:** Random agent and a simple rule-based heuristic (`jump if dx < k * speed`).
2. **Tabular RL (Q-Learning / SARSA):** Discretized feature state to demonstrate fundamentals and analyze where discretization breaks down (e.g., high speed, multi-obstacle sequences).
3. **Deep RL — Vector State (DQN / PPO):** Fast training, hyperparameter tuning, frame-skipping analysis, and reward-shaping ablations.
4. **Deep RL — Pixel State (CNN-DQN):** Sample efficiency vs. vector state, handling day/night inversion (color-inversion augmentation or frame differencing).
5. **Sim-to-Real Browser Transfer:** Evaluate Python-trained agents in real Chrome Dino via Playwright/Selenium and diagnose any sim-to-real performance gap.

---

## Deliverables Alignment (`final_project.pdf`)
- **Presentation (15 min + 5–10 min Q&A, ~June 22–26):** Problem setup, MDP design, failure modes & diagnostics, ablation results, and live browser demo.
- **Written Report (due 1 week after presentation):** Full technical details, experiment logs, and answers to Q&A questions.
