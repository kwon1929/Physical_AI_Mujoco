"""Config for G1Walk-v6 (V6h: pitch 강화 + height 조정 + 약한 lateral, 8M)."""
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" / "ppo_g1_v6h"
MODEL_DIR = PROJECT_ROOT / "models" / "ppo_g1_v6h"
VIDEO_DIR = PROJECT_ROOT / "videos"
MENAGERIE_DIR = PROJECT_ROOT / "mujoco_menagerie"

# ──────────────────────────────────────────────
# Environment (G1Walk-v6)
# ──────────────────────────────────────────────
ENV_ID = "G1Walk-v6"
N_ENVS = 4

# Reward (V6h = V6g-fix + pitch강화 + height조정 + 약한lateral)
FORWARD_REWARD_WEIGHT = 5.0        # exp 기반
FORWARD_SIGMA = 4.0                # exp 폭 축소 (1.0→4.0, 서있기 방지)
TARGET_VELOCITY = 0.5              # 목표 속도 0.5 m/s
UPRIGHT_REWARD_WEIGHT = 3.0        # 직립 보너스
UPRIGHT_SIGMA = 5.0                # roll 계수: exp(-5.0 * roll^2 - 10.0 * pitch^2)
UPRIGHT_PITCH_SIGMA = 10.0         # V6h: pitch 계수 2배 (5.0→10.0)
HEIGHT_REWARD_WEIGHT = 2.5         # V6h: 높이 가중치 상향 (2.0→2.5)
HEIGHT_SIGMA = 15.0                # V6h: 높이 감쇠 강화 (10.0→15.0)
TARGET_HEIGHT = 0.75               # V6h: 타겟 높이 상향 (0.73→0.75)
HEALTHY_REWARD = 0.5
CTRL_COST_WEIGHT = 0.001
# V6h: foot contact 유지 + 약한 lateral 재추가
SINGLE_FOOT_REWARD_WEIGHT = 1.0    # 한 발 접촉 보너스
CONTACT_THRESHOLD = 16.4           # 33.34kg * 9.81 * 0.05
LATERAL_VEL_COST_WEIGHT = 0.15     # V6h: 약하게 재추가 (V6g 0.3의 절반)
LATERAL_POS_COST_WEIGHT = 0.1      # V6h: 약하게 재추가 (V6g 0.2의 절반)

# Termination
HEALTHY_Z_RANGE = (0.5, 1.5)
MAX_ROLL_PITCH = 0.8               # ~45도

# Other
FRAME_SKIP = 5
ACTION_SCALE = 0.4                 # 축소 (0.7 → 0.4)

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
LOG_STD_INIT = -1.0
ORTHO_INIT = True

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
TOTAL_TIMESTEPS = 8_000_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 5
CHECKPOINT_FREQ = 1_000_000        # 1M마다 체크포인트 저장

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cpu"
