"""Config for G1Walk-v4 (Exponential Reward + Stability)."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1_v4"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1_v4"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (G1Walk-v4 - Exponential Reward)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v4"
N_ENVS = 4

# Target & Reward weights (지수 기반!)
TARGET_VELOCITY = 1.0            # 목표 속도 1.0 m/s
FORWARD_REWARD_WEIGHT = 10.0     # 전진 보상
STABILITY_REWARD_WEIGHT = 5.0    # 자세 안정성 보상 (NEW!)
HEALTHY_REWARD = 3.0             # 생존 보상
CTRL_COST_WEIGHT = 0.01          # 토크 비용

# Termination
HEALTHY_Z_RANGE = (0.25, 1.5)
MAX_ROLL_PITCH = 0.8             # 최대 roll/pitch (라디안, ~45도)

# Other
FRAME_SKIP = 5
ACTION_SCALE = 0.7

# ──────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
MAX_GRAD_NORM = 0.5
VF_COEF = 0.5
ENT_COEF = 0.01

# Network
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
LOG_STD_INIT = -0.5
ORTHO_INIT = True

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 4_000_000      # V3와 동일
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
SAVE_FREQ = 50_000

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cpu"
