"""Config for G1Walk-v6 (V5 + Upright Bonus - Ablation Step 1)."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1_v6"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1_v6"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (G1Walk-v6)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v6"
N_ENVS = 4

# Reward
FORWARD_REWARD_WEIGHT = 10.0       # 선형 forward (V5와 동일)
UPRIGHT_REWARD_WEIGHT = 3.0        # NEW: 직립 보너스
UPRIGHT_SIGMA = 5.0                # NEW: exp(-5.0 * (roll^2 + pitch^2))
HEALTHY_REWARD = 0.5               # 낮게 (V5와 동일)
CTRL_COST_WEIGHT = 0.001           # 최소 (V5와 동일)

# Termination (V5와 동일)
HEALTHY_Z_RANGE = (0.5, 1.5)
MAX_ROLL_PITCH = 0.8               # ~45도

# Other
FRAME_SKIP = 5
ACTION_SCALE = 0.7

# ──────────────────────────────────────────────
# PPO Hyperparameters (V5와 동일)
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
LOG_STD_INIT = -1.0
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
