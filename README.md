# Dino RL

The first milestone is a seeded, original dinosaur runner for controlled reinforcement learning experiments. See the [project plan](PROJECT_PLAN.md) for the DQN and PPO comparison.

## Run the environment

Requires Python 3.10 or newer.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
from agents.heuristic import timing_policy
from env import DinoEnv

env = DinoEnv()
obs, info = env.reset(seed=42)
while True:
    obs, reward, terminated, truncated, info = env.step(timing_policy(obs))
    if terminated or truncated:
        print(info)
        break
env.close()
```

Use `env.action_space.sample()` in place of `timing_policy(obs)` for a random policy. Seed the action space separately for repeatable random actions. Set `render_mode="rgb_array"` and call `env.render()` for a debug frame.

## Environment contract

- Actions are run (`0`), jump (`1`), and duck (`2`). Ducking in the air speeds up the fall. Each step advances one tick.
- Observations are 11 float32 features in this order: feet height, vertical velocity (positive upward), speed, then the signed gap, bottom height, width, and height of each of the two nearest uncleared obstacles. The signed gap is measured from the runner’s front to the obstacle’s front. Missing obstacles have four zeros. A fixed, monotonic normalization `v / (scale + abs(v))` preserves speeds above the nominal range.
- The reward is +1 for a survived tick and −10 on a crash. A crash terminates the episode. Surviving the time limit truncates it. Info includes elapsed ticks, survived ticks, distance, obstacles cleared, course seed, speed multiplier, and crash type.
- Configure base speed, acceleration, jump parameters, obstacle frequencies, gaps, and time cap through `DinoConfig`. Use `reset(seed=42, options={"speed_multiplier": 1.25})` to replay the same layout at higher speed. The entire speed curve scales while obstacle positions stay fixed. Pass `course_seed` separately to decouple the course from the reset seed. Pass an ordered tuple of `Obstacle` objects as `options["course"]` to replay a scripted layout, including held-out obstacle combinations.
- The feasibility test enumerates the reachable jump and duck states for representative nominal, accelerated, and shorter-gap courses. It uses full course knowledge. The timing policy only observes the current vector and serves as a separate baseline.

## Implementation notes

World positions are in distance units and speed is in units per tick. Course generation uses its own NumPy generator and samples width, type, and spacing independently of actions or speed. Keep the NumPy version with experiment logs. Rectangular hitboxes and original obstacle geometry make the simulator independent of browser assets and browser physics.

### Pilot checks

Gymnasium 1.4.0 and Python 3.11 passed the API and regression tests. A reachability check found a surviving action sequence for each of 120 pilot cases: 20 seeds (25–44), two gap ranges (95–190 and 91–145), and three speed multipliers (1.00, 1.15, and 1.25). An earlier test with gaps of 78–140 at 1.25× produced an unavoidable collision on seed 12. Recheck the final course list once its layout rules and seed splits are frozen.

Across the 20 nominal pilot layouts the timing heuristic averaged 352.45 survived ticks at nominal speed and 357.35 at 1.25×. The random policy averaged 24.85 and 19.85 ticks, respectively. These pilot checks establish that the baseline can clear obstacles and that the cap leaves headroom. Faster speed changes jump timing and can help on some layouts, so report each stress level on its own.
