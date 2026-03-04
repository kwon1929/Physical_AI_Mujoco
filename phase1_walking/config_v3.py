"""Config for G1Walk-v3 (Lean Walking - Simplified)."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1_v3"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1_v3"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (G1Walk-v3 - Lean Walking)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v3"
N_ENVS = 4

# Reward weights (단순화! 걷기에 집중)
FORWARD_REWARD_WEIGHT = 10.0   # 선형, 높게
CTRL_COST_WEIGHT = 0.001       # 최소화
HEALTHY_REWARD = 5.0           # 매 스텝 높은 보상
HEALTHY_Z_RANGE = (0.25, 1.5)  # 완화된 종료 조건
FRAME_SKIP = 5
ACTION_SCALE = 0.7             # 0.3 → 0.7 증가

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
ENT_COEF = 0.01              # 기본값으로 복구

# Network
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
LOG_STD_INIT = -0.5          # 초기 분산 유지
ORTHO_INIT = True

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 4_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
SAVE_FREQ = 50_000

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cpu"
