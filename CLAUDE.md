# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Humanoid Swarm Intelligence (HSI)** - A multi-phase MuJoCo humanoid robotics project progressing from single-robot walking to swarm construction.

- **Current Phase**: Phase 1 - Unitree G1 Walking (4 weeks)
- **Future Phases**: Task Execution (6w) → Multi-Agent (6w) → Swarm Construction (8w)
- **Robot Model**: Unitree G1 from MuJoCo Menagerie (29 DoF, position control)
- **RL Framework**: Stable-Baselines3 PPO + VecNormalize
- **Platform**: macOS arm64, Python 3.13

## Development Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Common Commands

### Week 1: Environment Exploration
```bash
# Test environment and view G1 standing
python -m phase1_walking.view_g1 --mode stand

# Explore environment details
python -m phase1_walking.env_test

# Compare humanoid models from Menagerie
python -m phase1_walking.playground_explore --model all
```

### Week 2-3: Training & Evaluation
```bash
# Start PPO training (~2-4 hours for 2M timesteps)
python -m phase1_walking.train

# Monitor training with TensorBoard
tensorboard --logdir logs/ppo_g1

# Evaluate trained model
python -m phase1_walking.evaluate
python -m phase1_walking.evaluate --render    # Live visualization
python -m phase1_walking.evaluate --record    # Save video to videos/
```

### Week 3: Hyperparameter Tuning
```bash
# Run tuning experiments
python -m phase1_walking.tune --experiment reward_weights
python -m phase1_walking.tune --experiment learning_rates
python -m phase1_walking.tune --experiment batch_sizes
```

## Architecture

### Core Components

**`phase1_walking/config.py`** - Centralized configuration hub
- All hyperparameters (PPO settings, reward weights, network architecture)
- Path definitions (logs, models, videos, menagerie)
- Environment parameters (G1-specific)
- Import this for any configuration values

**`phase1_walking/g1_env.py`** - Custom Gymnasium environment for Unitree G1
- Observation: 69 dims (qpos excluding x,y + qvel)
- Action: 29 dims (position offsets for actuators, scaled by `action_scale`)
- Reward: `forward_velocity * weight + healthy_bonus - ctrl_cost`
- Termination: pelvis height outside `HEALTHY_Z_RANGE`
- Reset: Uses "stand" keyframe from G1 model + small noise
- Registered as `"G1Walk-v0"`

**`phase1_walking/train.py`** - PPO training pipeline
- Creates vectorized environments: `SubprocVecEnv` (4 parallel) + `VecNormalize`
- **Critical**: Saves both `model.zip` AND `vec_normalize.pkl`
- Callbacks: EvalCallback (saves best model), ProgressCallback
- macOS: Uses `multiprocessing.set_start_method("fork")`

**`phase1_walking/evaluate.py`** - Model evaluation and video recording
- **Critical**: Must load `vec_normalize.pkl` with the model
- Functions: `run_evaluation()`, `record_video()`, `render_live()`
- Success criterion: mean episode length >= 100 steps

### Key Technical Details

**VecNormalize is mandatory**
- G1's 69-dim observation has vastly different scales
- Running statistics must be saved/loaded with the model
- Training: `norm_reward=True`, Eval: `norm_reward=False`
- Saved as `models/ppo_g1/vec_normalize.pkl`

**G1 Model Specifics**
- 29 actuators (6 per leg, 3 waist, 7 per arm)
- Position control with `kp=500` (not torque)
- "stand" keyframe provides initial stable pose
- Pelvis starts at z=0.79m in standing pose
- Total mass: 33.34 kg

**macOS-Specific**
- `SubprocVecEnv` requires `multiprocessing.set_start_method("fork", force=True)`
- MuJoCo viewer: Use `mujoco.viewer.launch()` not `launch_passive()`
- MPS device available but unstable with SB3 → CPU recommended

**File Organization**
```
phase1_walking/
├── config.py           # Single source of truth for all settings
├── g1_env.py          # Custom Gymnasium environment (registered as G1Walk-v0)
├── train.py           # PPO training (SubprocVecEnv + VecNormalize)
├── evaluate.py        # Evaluation + video recording
├── callbacks.py       # SB3 callbacks (EvalCallback, ProgressCallback)
├── tune.py            # Hyperparameter sweeps
├── view_g1.py         # Visualization without RL
├── env_test.py        # Environment exploration
└── playground_explore.py  # Menagerie model comparison

logs/ppo_g1/           # TensorBoard logs (gitignored)
models/ppo_g1/         # Model checkpoints + vec_normalize.pkl (gitignored)
videos/                # Evaluation videos (gitignored)
mujoco_menagerie/      # 70+ robot models including G1
```

## Modifying Training

**Changing hyperparameters**: Edit `config.py` (centralized)

**Custom reward function**: Modify `G1WalkEnv.step()` in `g1_env.py`

**Network architecture**: Update `NET_ARCH` in `config.py`

**Environment parameters**: Adjust `HEALTHY_Z_RANGE`, `action_scale`, etc. in `config.py` or pass to `gym.make()`

## Testing Changes

Always test with short timesteps first:
```python
# In train.py, temporarily change:
TOTAL_TIMESTEPS = 10_000  # Quick sanity check
```

Check that both model and VecNormalize are saved:
```bash
ls models/ppo_g1/
# Should see: best_model.zip, final_model.zip, vec_normalize.pkl
```

## Future Phases Structure

When implementing Phase 2-4:
- Create `phase2_task/`, `phase3_multiagent/`, `phase4_swarm/` directories
- Each phase has its own `config.py` for phase-specific settings
- Share the same `venv` and `mujoco_menagerie/`
- Phase 2+: Will need custom MuJoCo scenes with objects/multiple robots

## Project Vision

From PDF: Multi-phase progression toward swarm construction
1. **Phase 1** (current): Single G1 learns to walk
2. **Phase 2**: Walk + manipulate objects (pick, place, carry)
3. **Phase 3**: Multiple G1s coordinate via MAPPO + LLM task planner
4. **Phase 4**: Swarm of 4+ G1s collaboratively build structures
