"""Train G1 with v6 environment (V6h: pitch 강화 + height 조정 + 약한 lateral, 8M)."""
import multiprocessing

import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

from phase1_walking import config_v6 as config
from phase1_walking.callbacks import EvalByDistanceCallback, ProgressCallback
from phase1_walking.g1_env_v6 import G1WalkEnvV6  # noqa: F401 (register env)


def make_env(rank: int, seed: int = 0):
    """환경 생성 팩토리."""
    def _init():
        env = gym.make(
            config.ENV_ID,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            forward_sigma=config.FORWARD_SIGMA,
            target_velocity=config.TARGET_VELOCITY,
            upright_reward_weight=config.UPRIGHT_REWARD_WEIGHT,
            upright_sigma=config.UPRIGHT_SIGMA,
            upright_pitch_sigma=config.UPRIGHT_PITCH_SIGMA,
            height_reward_weight=config.HEIGHT_REWARD_WEIGHT,
            height_sigma=config.HEIGHT_SIGMA,
            target_height=config.TARGET_HEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            single_foot_reward_weight=config.SINGLE_FOOT_REWARD_WEIGHT,
            contact_threshold=config.CONTACT_THRESHOLD,
            lateral_vel_cost_weight=config.LATERAL_VEL_COST_WEIGHT,
            lateral_pos_cost_weight=config.LATERAL_POS_COST_WEIGHT,
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
    print(" G1 Walking Training V6g-fix")
    print(" V6f + Fixed Foot Contact, No Lateral (8M)")
    print(" EvalCallback: best by forward distance")
    print("=" * 60)
    print(f"\n  ENV: {config.ENV_ID}")
    print(f"  N_ENVS: {config.N_ENVS}")
    print(f"  TOTAL_TIMESTEPS: {config.TOTAL_TIMESTEPS:,}")
    print(f"  DEVICE: {config.DEVICE}")
    print(f"  ACTION_SCALE: {config.ACTION_SCALE}")
    print(f"\n  forward ({config.FORWARD_REWARD_WEIGHT} * exp(-{config.FORWARD_SIGMA} * (vel-{config.TARGET_VELOCITY})^2))")
    print(f"  + upright ({config.UPRIGHT_REWARD_WEIGHT} * exp(-{config.UPRIGHT_SIGMA} * angle^2))")
    print(f"  + height ({config.HEIGHT_REWARD_WEIGHT} * exp(-{config.HEIGHT_SIGMA} * (z-{config.TARGET_HEIGHT})^2))")
    print(f"  + single_foot ({config.SINGLE_FOOT_REWARD_WEIGHT}, threshold={config.CONTACT_THRESHOLD:.1f}N)")
    print(f"  + healthy ({config.HEALTHY_REWARD})")
    print(f"  - ctrl_cost ({config.CTRL_COST_WEIGHT})")
    print(f"  - lateral ({config.LATERAL_VEL_COST_WEIGHT}*vy^2 + {config.LATERAL_POS_COST_WEIGHT}*y^2)")
    print(f"  termination: z>{config.HEALTHY_Z_RANGE[0]}m, roll/pitch<{config.MAX_ROLL_PITCH}rad\n")

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
    eval_callback = EvalByDistanceCallback(
        eval_env=eval_env,
        model_dir=config.MODEL_DIR,
        log_dir=config.LOG_DIR,
        eval_freq=config.EVAL_FREQ,
        n_eval_episodes=config.N_EVAL_EPISODES,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config.CHECKPOINT_FREQ // config.N_ENVS,  # per-env steps
        save_path=str(config.MODEL_DIR / "checkpoints"),
        name_prefix="v6g_fix",
    )
    progress_callback = ProgressCallback(
        total_timesteps=config.TOTAL_TIMESTEPS,
        print_freq=10000,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback, progress_callback])

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
