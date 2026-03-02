"""Week 2: Train PPO on Humanoid-v5.

Run:     python -m phase1_walking.train
Monitor: tensorboard --logdir logs/ppo_humanoid
"""
import multiprocessing
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from phase1_walking import config
from phase1_walking.callbacks import make_eval_callback, ProgressCallback


def make_env(env_id: str, rank: int, seed: int = 0):
    """단일 환경 생성 팩토리 (SubprocVecEnv용).

    Args:
        env_id: Gymnasium 환경 ID
        rank: 환경 인덱스 (시드 오프셋)
        seed: 기본 시드
    """
    def _init():
        env = gym.make(
            env_id,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            healthy_z_range=config.HEALTHY_Z_RANGE,
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


def create_training_env(n_envs: int) -> VecNormalize:
    """벡터화 + 정규화된 학습 환경 생성.

    구조: gym.make() x N → SubprocVecEnv → VecNormalize

    VecNormalize는 Humanoid 학습에 필수!
    348차원 관측값의 스케일이 매우 다르기 때문에,
    정규화 없이는 학습이 거의 수렴하지 않음.
    """
    try:
        vec_env = SubprocVecEnv([make_env(config.ENV_ID, i) for i in range(n_envs)])
    except RuntimeError:
        # macOS fork 이슈 발생 시 DummyVecEnv로 폴백
        print("SubprocVecEnv 실패, DummyVecEnv로 전환합니다.")
        vec_env = DummyVecEnv([make_env(config.ENV_ID, i) for i in range(n_envs)])

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def create_eval_env() -> VecNormalize:
    """평가용 단일 환경 생성 (reward 정규화 OFF)."""
    eval_env = DummyVecEnv([make_env(config.ENV_ID, rank=99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return eval_env


def build_model(env: VecNormalize) -> PPO:
    """PPO 모델 생성 (config.py의 하이퍼파라미터 사용).

    핵심 설정:
        - MlpPolicy [256, 256] for pi and vf (RL Zoo 기반)
        - ReLU activation (Humanoid locomotion에 Tanh보다 효과적)
        - log_std_init=-2: 초기 액션 분산을 작게 (안정성 향상)
    """
    policy_kwargs = dict(
        net_arch=config.NET_ARCH,
        activation_fn=nn.ReLU,
        log_std_init=config.LOG_STD_INIT,
        ortho_init=config.ORTHO_INIT,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.LEARNING_RATE,
        n_steps=config.N_STEPS,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        max_grad_norm=config.MAX_GRAD_NORM,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(config.LOG_DIR),
        verbose=1,
        seed=42,
        device=config.DEVICE,
    )
    return model


def train():
    """전체 학습 파이프라인."""
    # 1. 디렉토리 생성
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 2. 환경 생성
    print(f"학습 환경 생성 중... (N_ENVS={config.N_ENVS})")
    train_env = create_training_env(config.N_ENVS)
    eval_env = create_eval_env()

    # 3. 모델 생성
    model = build_model(train_env)
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"모델 생성 완료:")
    print(f"  Device: {model.device}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Policy: {model.policy.__class__.__name__}")

    # 4. 콜백 설정
    eval_callback = make_eval_callback(
        eval_env,
        config.LOG_DIR,
        config.MODEL_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
    )
    progress_callback = ProgressCallback(
        print_freq=10_000,
        total_timesteps=config.TOTAL_TIMESTEPS,
    )

    # 5. 학습 시작
    print(f"\n{'=' * 60}")
    print(f" PPO Training: {config.ENV_ID}")
    print(f" Total timesteps: {config.TOTAL_TIMESTEPS:,}")
    print(f" TensorBoard: tensorboard --logdir {config.LOG_DIR}")
    print(f"{'=' * 60}\n")

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=[eval_callback, progress_callback],
        progress_bar=True,
    )

    # 6. 모델 + VecNormalize 저장
    model_path = config.MODEL_DIR / "final_model"
    normalize_path = config.MODEL_DIR / "vec_normalize.pkl"

    model.save(str(model_path))
    train_env.save(str(normalize_path))

    print(f"\n{'=' * 60}")
    print(f" Training complete!")
    print(f" Model:        {model_path}.zip")
    print(f" VecNormalize: {normalize_path}")
    print(f" TensorBoard:  tensorboard --logdir {config.LOG_DIR}")
    print(f"{'=' * 60}")

    # 7. 정리
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    # macOS에서 SubprocVecEnv를 위해 필요
    multiprocessing.set_start_method("fork", force=True)
    train()
