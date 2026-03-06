"""Train G1 with v5 environment (Linear Forward + Stability Termination)."""
import multiprocessing

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList

from phase1_walking import config_v5 as config
from phase1_walking.callbacks import make_eval_callback, ProgressCallback
from phase1_walking.g1_env_v5 import G1WalkEnvV5  # noqa: F401 (register env)


def make_env(rank: int, seed: int = 0):
    """환경 생성 팩토리."""
    def _init():
        env = gym.make(
            config.ENV_ID,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            healthy_z_range=config.HEALTHY_Z_RANGE,
            max_roll_pitch=config.MAX_ROLL_PITCH,
            action_scale=config.ACTION_SCALE,
            frame_skip=config.FRAME_SKIP,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    print("=" * 60)
    print(" G1 Walking Training v5 (Linear Forward + Stability Term)")
    print("=" * 60)
    print(f"\n  ENV: {config.ENV_ID}")
    print(f"  N_ENVS: {config.N_ENVS}")
    print(f"  TOTAL_TIMESTEPS: {config.TOTAL_TIMESTEPS:,}")
    print(f"  DEVICE: {config.DEVICE}")
    print(f"\n  V3 forward (linear * {config.FORWARD_REWARD_WEIGHT})")
    print(f"  + V4 termination (z>{config.HEALTHY_Z_RANGE[0]}m, roll/pitch<{config.MAX_ROLL_PITCH}rad)")
    print(f"  healthy_reward={config.HEALTHY_REWARD} (low)\n")

    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        multiprocessing.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # 환경 생성
    print("Creating environments...")
    env = SubprocVecEnv([make_env(i) for i in range(config.N_ENVS)])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=config.GAMMA,
    )

    eval_env = DummyVecEnv([make_env(0)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=config.GAMMA,
        training=False,
    )

    # 콜백
    eval_callback = make_eval_callback(
        eval_env=eval_env,
        log_dir=config.LOG_DIR,
        model_dir=config.MODEL_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
    )
    progress_callback = ProgressCallback(
        total_timesteps=config.TOTAL_TIMESTEPS,
        print_freq=10000,
    )
    callbacks = CallbackList([eval_callback, progress_callback])

    # PPO 모델
    print("Creating PPO model...")
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
            "net_arch": config.NET_ARCH,
            "activation_fn": nn.ReLU,
            "log_std_init": config.LOG_STD_INIT,
            "ortho_init": config.ORTHO_INIT,
        },
        tensorboard_log=str(config.LOG_DIR),
        device=config.DEVICE,
        verbose=1,
    )

    print(f"\nTensorBoard: tensorboard --logdir {config.LOG_DIR}")
    print("\nTraining started...\n")

    model.learn(
        total_timesteps=config.TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )

    # 저장
    model.save(str(config.MODEL_DIR / "final_model"))
    env.save(str(config.MODEL_DIR / "vec_normalize.pkl"))

    print("\n" + "=" * 60)
    print(" Training complete!")
    print("=" * 60)
    print(f"\n  Model: {config.MODEL_DIR}/")
    print(f"  Logs:  {config.LOG_DIR}/\n")

    env.close()


if __name__ == "__main__":
    main()
