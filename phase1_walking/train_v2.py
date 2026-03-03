"""Train G1 with v2 environment (improved reward shaping)."""
import multiprocessing
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList

from phase1_walking import config_v2 as config
from phase1_walking.callbacks import make_eval_callback, ProgressCallback
from phase1_walking.g1_env_v2 import G1WalkEnvV2  # Import to register


def make_env(rank: int, seed: int = 0):
    """환경 생성 팩토리."""
    def _init():
        env = gym.make(
            config.ENV_ID,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            action_rate_weight=config.ACTION_RATE_WEIGHT,
            energy_weight=config.ENERGY_WEIGHT,
            foot_contact_reward=config.FOOT_CONTACT_REWARD,
            contact_consistency_weight=config.CONTACT_CONSISTENCY_WEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            healthy_z_range=config.HEALTHY_Z_RANGE,
            action_scale=config.ACTION_SCALE,
            frame_skip=config.FRAME_SKIP,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("=" * 60)
    print(" 🚀 G1 Walking Training v2 (Improved Reward Shaping)")
    print("=" * 60)
    print(f"\n📦 환경: {config.ENV_ID}")
    print(f"🔧 병렬 환경 수: {config.N_ENVS}")
    print(f"🎯 총 학습 스텝: {config.TOTAL_TIMESTEPS:,}")
    print(f"💻 디바이스: {config.DEVICE}\n")

    # 디렉토리 생성
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # macOS에서 SubprocVecEnv 사용 시 필요
    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # 환경 생성
    print("🏗️  환경 생성 중...")
    env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.GAMMA,
    )

    # 콜백 설정
    eval_callback = make_eval_callback(
        env_id=config.ENV_ID,
        log_dir=config.LOG_DIR,
        model_dir=config.MODEL_DIR,
        n_eval_episodes=config.N_EVAL_EPISODES,
        eval_freq=config.EVAL_FREQ,
        deterministic=True,
        render=False,
        verbose=1,
    )

    progress_callback = ProgressCallback(
        total_timesteps=config.TOTAL_TIMESTEPS,
        check_freq=1000,
    )

    callbacks = CallbackList([eval_callback, progress_callback])

    # PPO 모델 생성
    print("\n🤖 PPO 모델 생성 중...")
    model = PPO(
        "MlpPolicy",
        env,
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
        policy_kwargs={
            "net_arch": [dict(**config.NET_ARCH)],
            "activation_fn": __import__("torch.nn", fromlist=["ReLU"]).ReLU,
            "log_std_init": config.LOG_STD_INIT,
            "ortho_init": config.ORTHO_INIT,
        },
        tensorboard_log=str(config.LOG_DIR),
        device=config.DEVICE,
        verbose=1,
    )

    print(f"\n📊 TensorBoard: tensorboard --logdir {config.LOG_DIR}")
    print("\n🎓 학습 시작...\n")

    # 학습
    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # 최종 모델 저장
    final_model_path = config.MODEL_DIR / "final_model"
    model.save(str(final_model_path))
    env.save(str(config.MODEL_DIR / "vec_normalize.pkl"))

    print("\n" + "=" * 60)
    print(" ✅ 학습 완료!")
    print("=" * 60)
    print(f"\n💾 모델 저장 위치: {config.MODEL_DIR}/")
    print(f"📊 로그 위치: {config.LOG_DIR}/")
    print(f"\n🎬 평가 실행:")
    print(f"   python -m phase1_walking.evaluate_v2 --render")
    print(f"   python -m phase1_walking.evaluate_v2 --record\n")

    env.close()


if __name__ == "__main__":
    main()
