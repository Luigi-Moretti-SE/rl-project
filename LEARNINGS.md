# Learnings & Research Notes

Persistent log of architectural decisions, prior-art research, and empirical findings so context is preserved across sessions.

---

## 1. Prior Art & Ecosystem Findings (Chrome Dino RL)

### Existing Approaches on GitHub
- **Browser-Wrapped Gym (`elvisyjlin/gym-chrome-dino`):** Wraps `chrome://dino` via Selenium/ChromeDriver + OpenCV. Suffers from WebDriver IPC latency and real-time FPS bottlenecks during training.
- **Pure-Python / Pygame Clones (`aome510/chrome-dino-game-rl`, `thomas-mauran/DinoAI`, `19-ade/gym-dino-game`):** Fast `gymnasium` training with DQN / Double DQN / PPO (PyTorch or Stable-Baselines3), but typically omit sim-to-real validation against the actual Chromium JS engine and often simplify physics.
- **In-Browser JS (`Runner.instance_` + TensorFlow.js):** Hooks directly into `Runner.instance_.tRex` and `Runner.instance_.horizon.obstacles` for zero-latency inference, but training inside a browser tab is slow.

### Why Our Hybrid Architecture Works Better
- **Train in Python (`env/` + `train.py`):** Headless `gymnasium` environment at thousands of FPS with deterministic seeds.
- **Deploy in Chrome (`extension/`):** Manifest V3 Chrome Extension running exported policies (JSON weights / ONNX) at 60 FPS inside a locally hosted copy of Chromium's `runner.js`.
- **Important Chrome Restriction:** Extensions cannot inject content scripts into `chrome://dino` (`chrome://*` URLs are blocked). Hosting the official Chromium T-Rex Runner HTML/JS locally (`http://localhost`) gives identical physics with full extension support.

---

## 2. Key Technical Pitfalls & Design Rules

1. **Sim-to-Real Physics Fidelity:**
   - Copy exact constants from Chromium's `runner.js` (gravity, initial jump velocity, fast-fall speed, acceleration curve, max speed, obstacle hitboxes, and 3 pterodactyl spawn heights: low / mid / high) into `env/` to minimize the sim-to-real gap.
2. **Day/Night Color Inversion (Pixel Mode):**
   - At score 700, Chrome Dino flips black-on-white ↔ white-on-black, causing naive CNN-DQN agents to crash immediately.
   - *Mitigation:* Detect background color (e.g., corner pixel) and invert dark frames during preprocessing, or train with random color-inversion augmentation.
3. **Frame Stacking & Frame Skipping:**
   - Single frames hide velocity/acceleration; stack 4 consecutive grayscale frames (`4 × 84 × 84`).
   - Frame-skip (`3–4`) speeds up training, but too large a frame-skip causes late reactions at high game speeds.
4. **State & Reward Design:**
   - Include the **next 2 obstacles** in the vector state so the agent doesn't jump over obstacle #1 directly into obstacle #2.
   - Pure `+1` per frame can encourage "bunny-hopping"; test small jump penalties (`-0.05`) or obstacle-clear bonuses as ablations.

---

## 3. Deep Research Findings (Chromium `offline.js`, RL Tuning, MV3 Extension)

### A. Exact Chromium `offline.js` Physics & Collision Spec
From Chromium's `Runner.config` and `Trex.collisionBoxes`:
- **Speed & Acceleration:**
  - `SPEED = 6` (initial speed)
  - `MAX_SPEED = 13`
  - `ACCELERATION = 0.001` (added per frame to `currentSpeed`)
  - `GAP_COEFFICIENT = 0.6` (controls minimum safe gap between obstacles as speed scales)
  - `INVERT_DISTANCE = 700` (triggers day/night color inversion every 700 units)
- **T-Rex Jump & Gravity:**
  - `GRAVITY = 0.6`
  - `INITIAL_JUMP_VELOCITY = 12` (upward thrust on jump)
  - `SPEED_DROP_COEFFICIENT = 3` (fast-fall multiplier when pressing Down in mid-air)
- **Two-Tier Hitbox Collision (`checkForCollision`):**
  - Broad outer bounding box check first, followed by fine-grained `CollisionBox(x, y, w, h)` checks:
  - `Trex.collisionBoxes.RUNNING` (6 boxes): `(22, 0, 17, 16)`, `(1, 18, 30, 9)`, `(10, 35, 14, 8)`, `(1, 24, 29, 5)`, `(5, 30, 21, 4)`, `(9, 34, 15, 4)`
  - `Trex.collisionBoxes.DUCKING` (1 box): `(1, 18, 55, 25)`
- **Obstacle Types:** `CACTUS_SMALL`, `CACTUS_LARGE`, `PTERODACTYL` (spawns after a score/speed threshold at 3 `yPos` altitudes: low, mid, high).

### B. RL Algorithms, Normalization & Hyperparameter Baselines
- **Vector State Normalization:** Min-max scale continuous features (`dx`, `width`, `height`, `y`, `speed`, `dino_y`, `dino_vy`) to `[0, 1]` or `[-1, 1]` so gradients stay well-conditioned as distances span `0–600+` px.
- **DQN / Double DQN Baseline:**
  - `lr = 1e-4` to `5e-4` (Adam), `gamma = 0.99`, `batch_size = 32` or `64`, replay buffer `50k–100k`, target update every `1,000–2,500` steps, epsilon decay `1.0 -> 0.02` over `20k–50k` steps.
- **PPO Baseline (Stable-Baselines3 / VecEnv):**
  - `lr = 3e-4`, `gamma = 0.99`, `gae_lambda = 0.95`, `clip_range = 0.2`, `n_steps = 2048`, `batch_size = 64`, `n_epochs = 10`, `ent_coef = 0.01` (keeps exploration alive early on).
- **Stability Tricks:**
  - Reward scaling/clipping (e.g., `[-1, 1]` or `+0.1`/`+1.0` alive, `-10`/`-50` crash).
  - Gradient norm clipping (`0.5` or `1.0`).
  - Always pass `game_speed` in vector observations so the policy adapts jump timing as speed scales from `6` to `13`.

### C. Manifest V3 Chrome Extension Architecture
- **The MV3 `world: "MAIN"` Pattern:**
  - By default, MV3 content scripts run in an `ISOLATED` world and **cannot** see `window.Runner.instance_`.
  - Setting `"world": "MAIN"` in `manifest.json` under `content_scripts` executes the agent script directly in the page's JS context so it can read `Runner.instance_.tRex` and `Runner.instance_.horizon.obstacles` at 60 FPS with zero overhead.
  - *Tradeoff:* `MAIN` world scripts cannot call `chrome.*` APIs directly. If the popup/UI needs to switch policies or toggle the agent, pair a `MAIN` world runner script with an `ISOLATED` world bridge using `window.postMessage()`.

---

## 4. Adopted Simulation Environment (`seeded-dino-runner` / `env/dino.py`)

Following `PROJECT_PLAN.md`, the team adopted the custom deterministic Gymnasium runner from `origin/seeded-dino-runner` (`env/dino.py`) as the core simulator for the 2×2 `{DQN, PPO} × {Nominal, Varied}` generalization study (keeping browser transfer as an optional post-experiment stretch goal):

- **Runner & Physics (`DinoConfig` defaults):**
  - Hitboxes: `RUNNER_WIDTH = 19`, `STANDING_HEIGHT = 29`, `DUCK_HEIGHT = 13` (swept horizontal collision per tick).
  - Dynamics: `initial_speed = 5.5`, `max_speed = 10.0`, `acceleration = 0.0035`, `jump_velocity = 15`, `gravity = 2`, `fast_fall_velocity = 8`, `max_steps = 1500`.
- **5 Hazard Types (`OBSTACLE_TYPES`, nominal weights `(4, 2, 1, 2, 1)`, `gap_range = (95, 190)`):**
  - `rock` (`bottom=0, h=21, w∈[17,34]`) & `thorn` (`bottom=0, h=29, w∈[13,23]`): ground hazards (must jump).
  - `skimmer` (`bottom=4, h=12, w∈[17,27]`): low flyer (`bottom < DUCK_HEIGHT`, must jump).
  - `glider` (`bottom=19, h=14, w∈[23,32]`): mid flyer (`DUCK_HEIGHT <= bottom < STANDING_HEIGHT`, duck under).
  - `soarer` (`bottom=39, h=15, w∈[20,31]`): high flyer (`bottom >= STANDING_HEIGHT`, safe to run under; jumping collides).
- **11-Feature Softsign Observation (`v / (scale + |v|)`):**
  - Order: `[height, velocity, speed, gap_1, bottom_1, width_1, height_1, gap_2, bottom_2, width_2, height_2]` (`0.0` for missing obstacles).
  - `OBSERVATION_SCALES = [64, 16, 10, 160, 48, 32, 32, 160, 48, 32, 32]` — preserves monotonicity beyond nominal `max_speed` during `1.15×` and `1.25×` speed-shift evaluations.
- **Reachability Constraint:**
  - `gap_range = (78, 140)` at `1.25×` speed produces unavoidable collisions (e.g., `course_seed=12`), whereas `(95, 190)` and `(91, 145)` passed full-course reachability (`course_is_playable`) across `1.00×`, `1.15×`, and `1.25×`. Always validate frozen train/dev/eval course splits with `course_is_playable` before training.
- **Sequence Holdout & Missing-Obstacle Invariants (2026-11-27):**
  - Default `generate_course()` samples obstacles i.i.d. from `OBSTACLE_TYPES`, meaning every consecutive pair/triple can appear in training unless explicitly filtered out via rejection sampling for the `Held-out obstacle sequences` split.
  - Because `DinoEnv.reset()` populates generated courses past `max_distance + RUNNER_WIDTH`, `obs[7:11] = [0, 0, 0, 0]` never appears mid-episode during training; any scripted `options["course"]` must likewise span past `max_steps * max_speed * speed_multiplier + RUNNER_WIDTH`. Under softsign normalization `v / (scale + |v|)`, `gap_2 = 0.0` is ambiguous with a 2nd obstacle immediately at the runner's front, which would also require sentinel padding or adaptation before any optional Chrome transfer (`horizon.obstacles` often has length 1).
