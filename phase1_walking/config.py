"""Centralized configuration for Phase 1: Unitree G1 walking."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (Unitree G1)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v0"
N_ENVS = 4  # 병렬 환경 수

# Reward weights (극단적 조정: 걷기 강제 유도)
FORWARD_REWARD_WEIGHT = 25.0  # 1.25 → 25.0 (20배! 전진 최우선)
CTRL_COST_WEIGHT = 0.1        # 유지
HEALTHY_REWARD = 0.5          # 5.0 → 0.5 (10배 감소, 서있기만 하면 손해)
HEALTHY_Z_RANGE = (0.3, 1.2)  # G1 pelvis 높이 범위 (stand ~0.79m)
FRAME_SKIP = 5                # Humanoid-v5와 동일 (1 step = 5 physics steps)

# ──────────────────────────────────────────────
# PPO Hyperparameters
# ──────────────────────────────────────────────
LEARNING_RATE = 3e-4
N_STEPS = 2048       # 환경당 rollout 스텝 수
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
MAX_GRAD_NORM = 0.5
VF_COEF = 0.5
ENT_COEF = 0.01     # 탐색 장려

# Network architecture
NET_ARCH = dict(pi=[256, 256], vf=[256, 256])
LOG_STD_INIT = -1.0   # G1은 position control이라 초기 분산 약간 크게
ORTHO_INIT = True

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 2_000_000
EVAL_FREQ = 10_000      # N 스텝마다 평가
N_EVAL_EPISODES = 5
SAVE_FREQ = 50_000      # N 스텝마다 체크포인트

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cpu"  # SB3의 MPS 지원이 불안정하므로 CPU 사용
