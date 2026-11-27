# Team Contract: Dependencies, Splits, Policy Loading & Eval Schemas

Short shared specification for **Member 1** (`env/`, `splits/`, `eval/`), **Member 2** (DQN & crash diagnostics), and **Member 3** (PPO, plots & presentation) so training, evaluation, and plotting plug together cleanly without cross-blocking.

---

## 1. Locked Python Stack (`requirements.txt`)

- `gymnasium>=1.2,<2`
- `numpy>=1.24,<3`
- `torch>=2.2,<3`
- `matplotlib>=3.8,<4`

PPO can be written in pure PyTorch (matching DQN) or wrapped from another library into the framework-agnostic `PolicyFn` callable below.

---

## 2. Policy Interface & Checkpoint Loading (`eval/` ↔ `agents/`)

Every baseline and learned policy consumed by `eval/` must match `agents.base.PolicyFn`:

```python
PolicyFn = Callable[[np.ndarray], int | Action]
```

- Input: `(11,)` `float32` softsign observation from `DinoEnv` (`height, velocity, speed, gap1, bottom1, width1, height1, gap2, bottom2, width2, height2`).
- Output: deterministic integer action `0` (`RUN`), `1` (`JUMP`), or `2` (`DUCK`).
- `agents/base.py` intentionally avoids importing `torch` so `eval/` and plotting scripts stay framework-agnostic.
- Algorithm loaders:
  - DQN: `agents.dqn.load_dqn_policy(checkpoint_path, device="cpu") -> PolicyFn`
  - PPO (Member 3): provide a matching `load_ppo_policy(checkpoint_path, device="cpu") -> PolicyFn`.

---

## 3. Course Manifest Contract (`splits/manifest.json`)

Member 1 freezes disjoint course seeds and held-out obstacle sequences in `splits/manifest.json`:

```json
{
  "held_out_transitions": [
    ["rock", "glider"],
    ["thorn", "rock"],
    ["skimmer", "thorn"],
    ["glider", "soarer"]
  ],
  "train_seeds": [10000, 10001, "..."],
  "dev_seeds": [2000, 2001, "..."],
  "eval_nominal_seeds": [3000, 3001, "..."],
  "eval_sequence_courses": [
    {
      "course_seed": 9000,
      "obstacles": [
        {"x": 155.0, "bottom": 0, "width": 20, "height": 21, "kind": "rock"}
      ]
    }
  ]
}
```

Notes:
- `dev_seeds`: 50 seeds.
- `eval_nominal_seeds`: 50 shared seeds reused across `held_out_nominal` (`1.0×`), `speed_1.15x`, and `speed_1.25x`.
- `eval_sequence_courses`: every course must span past `1500 * 10.0 * 1.25 + 19 = 18,769` units so `obs[7:11]` never drops to `[0, 0, 0, 0]` mid-episode.
- Before `splits/manifest.json` exists, `agents/course_sampler.py` falls back to disjoint pilot seeds (`train: 10000..10999`, `pilot_dev: 2000..2049`) and tags logs with `"split": "pilot_dev"`.

---

## 4. Standardized Training & Evaluation JSON Schemas

### A. Dev Learning Curve Log (`<algo>_<regime>_seed<N>_dev.json`)
Logged every `25,000` environment ticks up to `300,000`:

```json
{
  "algo": "dqn",
  "regime": "nominal",
  "train_seed": 0,
  "total_timesteps": 300000,
  "eval_interval": 25000,
  "split": "dev",
  "hyperparameters": {},
  "versions": {"python": "...", "numpy": "...", "gymnasium": "...", "torch": "..."},
  "checkpoints": [
    {
      "step": 25000,
      "mean_survival_ticks": 120.4,
      "std_survival_ticks": 45.2,
      "cap_reach_rate": 0.0,
      "mean_cleared_obstacles": 4.8,
      "wall_time_sec": 6.21
    }
  ]
}
```

### B. Final Evaluation Log (`<algo>_<regime>_seed<N>_eval.json`)
Produced by running a frozen policy on the shared `eval/` harness across the four suites (`held_out_nominal`, `speed_1.15x`, `speed_1.25x`, `held_out_sequences`):

```json
{
  "algo": "dqn",
  "regime": "varied",
  "train_seed": 0,
  "suites": {
    "held_out_nominal": {
      "mean_survival_ticks": 410.2,
      "std_survival_ticks": 180.5,
      "cap_reach_rate": 0.12,
      "crash_counts_by_pair": {"rock->glider": 0, "thorn->rock": 3},
      "crash_tag_counts": {"late_jump": 15, "missed_duck": 8},
      "episodes": [
        {
          "course_seed": 3000,
          "speed_multiplier": 1.0,
          "survival_ticks": 245,
          "truncated": false,
          "cleared_obstacles": 11,
          "distance": 1540.2,
          "crash_type": "glider",
          "prev_obstacle": "rock",
          "crash_tag": "missed_duck"
        }
      ]
    }
  }
}
```
