# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Humanoid Swarm Intelligence (HSI)** - A multi-phase MuJoCo humanoid robotics project progressing from single-robot walking to swarm construction.

- **Current Phase**: Phase 1 - Unitree G1 Walking (in progress, v6 iteration)
- **Future Phases**: Task Execution (6w) → Multi-Agent (6w) → Swarm Construction (8w)
- **Robot Model**: Unitree G1 from MuJoCo Menagerie (29 DoF, position control)
- **RL Framework**: Stable-Baselines3 PPO + VecNormalize
- **Platform**: macOS arm64 (primary), Linux compatible, Python 3.13

## Repository Structure

```
Physical_AI_Mujoco/
├── phase1_walking/              # Core RL module
│   ├── config.py                # v1 config (single source of truth)
│   ├── config_v2.py–v6.py       # Versioned configs (reward shaping experiments)
│   ├── g1_env.py                # v1 Gymnasium environment (G1Walk-v0)
│   ├── g1_env_v2.py–v6.py       # Iterative environment improvements
│   ├── train.py                 # v1 PPO training pipeline
│   ├── train_v2.py–v6.py        # Version-matched training scripts
│   ├── evaluate.py              # v1 evaluation & video recording
│   ├── evaluate_v2.py–v4.py     # Version-matched evaluators
│   ├── callbacks.py             # SB3 callbacks (shared across versions)
│   ├── tune.py                  # Hyperparameter sweep framework
│   ├── view_g1.py               # G1 visualization without RL
│   ├── env_test.py              # Environment exploration & validation
│   ├── playground_explore.py    # MuJoCo Menagerie model comparison
│   └── __init__.py              # Registers G1Walk-v0 on import
│
├── daily_update.py              # Auto: stats collection, video, GitHub commit
├── debug_agent.py               # Inspect learned policy action statistics
├── create_vec_normalize.py      # Regenerate vec_normalize.pkl if missing
├── watch_g1.py                  # Real-time agent viewer (trained or random)
├── view_trained_g1.py           # Model visualization wrapper
├── view_live.py–v6_interactive.py  # Progressive live viewer iterations
├── setup_auto_update.sh         # Cron job setup for daily_update.py
│
├── CLAUDE.md                    # This file
├── README.md                    # Minimal readme
├── REWARD_SHAPING_GUIDE.md      # Comprehensive reward design strategies
├── V0_VS_V2_COMPARISON.md       # Version diff documentation
├── V2_VS_V3_COMPARISON.md
├── V3_VS_V4_COMPARISON.md
│
├── requirements.txt             # Python dependencies
├── .env.example                 # Twitter/X API credentials template
├── humanoid_swarm_project.pdf   # Original project specification
├── draft_posts/                 # Auto-generated Twitter draft posts
│
├── logs/ppo_g1/                 # TensorBoard logs (gitignored)
├── models/ppo_g1/               # Model checkpoints (gitignored)
├── videos/                      # Recorded evaluation videos (gitignored)
└── mujoco_menagerie/            # 70+ robot models including G1
```

## Development Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Common Commands

### Environment Exploration (Week 1)
```bash
# View G1 in standing pose
python -m phase1_walking.view_g1 --mode stand

# Explore observation/action space
python -m phase1_walking.env_test

# Compare humanoid models from Menagerie
python -m phase1_walking.playground_explore --model all
```

### Training & Evaluation (Week 2–3)
```bash
# Start PPO training (~2–4 hours for 2M timesteps)
python -m phase1_walking.train

# Run versioned training (replace N with 2–6)
python -m phase1_walking.train_vN

# Monitor with TensorBoard
tensorboard --logdir logs/ppo_g1

# Evaluate trained model
python -m phase1_walking.evaluate
python -m phase1_walking.evaluate --render    # Live MuJoCo viewer
python -m phase1_walking.evaluate --record    # Save video to videos/
```

### Debugging & Visualization
```bash
# Watch trained agent live (or random for comparison)
python watch_g1.py
python watch_g1.py --random

# Inspect learned action statistics
python debug_agent.py

# Recreate missing vec_normalize.pkl
python create_vec_normalize.py
```

### Hyperparameter Tuning (Week 3)
```bash
python -m phase1_walking.tune --experiment reward_weights
python -m phase1_walking.tune --experiment learning_rates
python -m phase1_walking.tune --experiment batch_sizes
```

### Automation
```bash
# Run daily update (stats + video + GitHub commit)
python daily_update.py

# Set up cron automation
bash setup_auto_update.sh
```

## Architecture

### Core Components

**`phase1_walking/config.py`** — Centralized configuration hub
- All hyperparameters: PPO settings, reward weights, network architecture
- Path definitions: LOG_DIR, MODEL_DIR, VIDEO_DIR, MENAGERIE_DIR
- Environment parameters: HEALTHY_Z_RANGE, action_scale, FRAME_SKIP
- **Always import from here** — never hardcode paths or hyperparameters

**`phase1_walking/g1_env.py`** — Custom Gymnasium environment (`G1Walk-v0`)
- **Observation**: 69 dims = `qpos[2:]` (34) + `qvel` (35)
  - Excludes global x,y position (translation-invariant)
  - Includes pelvis z-height + all 29 joint angles + full velocity
- **Action**: 29 dims (position control offsets, scaled by `action_scale=0.3`, clipped to `[-1, 1]`)
- **Reward**: `forward_velocity * weight + healthy_bonus - ctrl_cost`
  - `forward_reward_weight = 25.0` (deliberately high to force walking)
  - `healthy_reward = 0.5`, `ctrl_cost_weight = 0.1`
- **Termination**: pelvis height outside `HEALTHY_Z_RANGE = (0.3, 1.2)`
- **Reset**: "stand" keyframe + Gaussian noise (σ=0.01)
- **Frame skip**: 5 physics steps per env step

**`phase1_walking/train.py`** — PPO training pipeline
- SubprocVecEnv (4 parallel) + VecNormalize wrapper
- **Critical**: Always saves both `model.zip` AND `vec_normalize.pkl`
- EvalCallback every 10,000 steps (5 episodes), saves best model
- macOS: `multiprocessing.set_start_method("fork", force=True)`

**`phase1_walking/evaluate.py`** — Model evaluation and video recording
- **Critical**: Must load `vec_normalize.pkl` alongside model weights
- `run_evaluation()` — deterministic rollout (default 10 episodes)
- `record_video()` — 1000-step recording saved to `videos/`
- `render_live()` — real-time MuJoCo viewer
- Success criterion: mean episode length ≥ 100 steps

**`phase1_walking/callbacks.py`** — SB3 callbacks (shared across all versions)
- `ProgressCallback` — prints timesteps, progress%, reward, FPS, ETA
- `make_eval_callback()` — factory for EvalCallback with best-model saving

**`phase1_walking/tune.py`** — Hyperparameter sweep framework
- `@dataclass Experiment`: name, description, config_overrides, timesteps
- Experiments: `reward_weights`, `learning_rates`, `batch_sizes`, `all`
- Logs all runs to TensorBoard for comparison

### Versioned Iteration Files

The codebase uses explicit version suffixes (v2–v6) to track reward shaping experiments. Each version is a self-contained experiment:

| Version | Key Change | Reference Doc |
|---------|-----------|---------------|
| v1 (base) | Forward reward=25.0, basic dense reward | — |
| v2 | Improved reward shaping (Tanh normalization) | V0_VS_V2_COMPARISON.md |
| v3 | Lean walking optimization | V2_VS_V3_COMPARISON.md |
| v4 | Exponential forward reward (Gaussian penalty around target velocity 1.0 m/s) | V3_VS_V4_COMPARISON.md |
| v5–v6 | Additional stability/contact rewards | — |

**Convention**: Each version N has matching `config_vN.py`, `g1_env_vN.py`, `train_vN.py`, and (where applicable) `evaluate_vN.py`. Always use matching versions together.

## Key Technical Details

### VecNormalize is Mandatory
- G1's 69-dim observation spans vastly different scales (meters, radians, m/s)
- Running statistics (mean/variance) must be saved with the model
- **Training**: `norm_reward=True` | **Evaluation**: `norm_reward=False`
- Saved as `models/ppo_g1/vec_normalize.pkl`
- If missing, regenerate with `python create_vec_normalize.py`

### G1 Robot Specifics
- 29 actuators: 6 per leg, 3 waist, 7 per arm
- Position control with `kp=500` (not torque control)
- "stand" keyframe used for initialization
- Pelvis z ≈ 0.79m in standing pose
- Total mass: 33.34 kg

### PPO Hyperparameters (v1 defaults)
```python
learning_rate = 3e-4
n_steps = 2048        # Per environment per rollout
batch_size = 256
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_grad_norm = 0.5
```

### Network Architecture
```python
NET_ARCH = [256, 256]  # Both policy and value function
```

### macOS-Specific Notes
- `SubprocVecEnv` requires `multiprocessing.set_start_method("fork", force=True)`
- MuJoCo viewer: `mujoco.viewer.launch()` — do NOT use `launch_passive()`
- MPS device available but unstable with SB3 → use CPU

### Reward Design Pitfall
The base `forward_reward_weight=25.0` is extreme to force walking behavior but can cause reward explosion and instability. See `REWARD_SHAPING_GUIDE.md` for 6 alternative strategies (Tanh normalization, exponential, sparse+dense, etc.). The v2–v6 environments each try different approaches.

## Expected Model Outputs

After training, verify all three files exist:
```bash
ls models/ppo_g1/
# best_model.zip       — weights at highest eval reward
# final_model.zip      — last checkpoint
# vec_normalize.pkl    — running observation statistics (REQUIRED for eval)
```

## Modifying Training

| Goal | Action |
|------|--------|
| Change hyperparameters | Edit `config.py` (or the matching `config_vN.py`) |
| Modify reward function | Edit `G1WalkEnv.step()` in `g1_env.py` |
| Change network size | Update `NET_ARCH` in `config.py` |
| Adjust env parameters | Edit `HEALTHY_Z_RANGE`, `action_scale`, `FRAME_SKIP` in config |
| Quick sanity check | Set `TOTAL_TIMESTEPS = 10_000` temporarily in train.py |

## Automation Infrastructure

**`daily_update.py`** — Scheduled automation script:
1. Collects latest training stats from `evaluations.npz`
2. Records a demo video via `evaluate --record`
3. Generates a Twitter/X post draft in `draft_posts/`
4. Commits progress to GitHub with training metrics

Setup requires Twitter API credentials in `.env` (copy from `.env.example`).

## Future Phases Structure

When implementing Phase 2–4:
- Create `phase2_task/`, `phase3_multiagent/`, `phase4_swarm/` directories
- Each phase has its own `config.py` for phase-specific settings
- Share the same `venv` and `mujoco_menagerie/`
- Phase 2+: Custom MuJoCo scenes with objects and multiple robots
- Phase 3: MAPPO algorithm + LLM-based task planner
- Phase 4: Swarm coordination for 4+ G1 robots

## Project Vision

Multi-phase progression toward swarm construction:
1. **Phase 1** (current): Single G1 learns to walk
2. **Phase 2**: Walk + manipulate objects (pick, place, carry)
3. **Phase 3**: Multiple G1s coordinate via MAPPO + LLM task planner
4. **Phase 4**: Swarm of 4+ G1s collaboratively build structures
