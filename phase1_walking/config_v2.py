"""Config for G1Walk-v2 (Improved Reward Shaping)."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1_v2"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1_v2"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (G1Walk-v2)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v2"
N_ENVS = 4

# Reward weights (Normalized & Balanced)
FORWARD_REWARD_WEIGHT = 2.0      # Tanh normalized → 낮춰도 OK
CTRL_COST_WEIGHT = 0.01
ACTION_RATE_WEIGHT = 0.1         # NEW: Smoothness
ENERGY_WEIGHT = 0.001            # NEW: Efficiency
FOOT_CONTACT_REWARD = 0.5        # NEW: Contact stability
CONTACT_CONSISTENCY_WEIGHT = 0.3 # NEW: Alternating gait
HEALTHY_REWARD = 0.2             # 더 낮춤
HEALTHY_Z_RANGE = (0.3, 1.2)
FRAME_SKIP = 5
ACTION_SCALE = 0.5               # 0.3 → 0.5 증가

# ──────────────────────────────────────────────
# PPO Hyperparameters (Exploration 증가)
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
ENT_COEF = 0.03              # 0.01 → 0.03 (더 많은 탐색!)

# Network
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
LOG_STD_INIT = -0.5          # -1.0 → -0.5 (초기 분산 증가)
ORTHO_INIT = True

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
SAVE_FREQ = 50_000

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cpu"
