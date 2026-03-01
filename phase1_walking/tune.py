"""Week 3: Hyperparameter tuning experiments.

Run: python -m phase1_walking.tune --experiment reward_weights
     python -m phase1_walking.tune --experiment learning_rates
     python -m phase1_walking.tune --experiment batch_sizes
     python -m phase1_walking.tune --experiment all

Compare: tensorboard --logdir logs/ppo_humanoid
"""
import argparse
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase1_walking import config


@dataclass
class Experiment:
    """단일 학습 실험 설정."""
    name: str
    description: str
    config_overrides: dict = field(default_factory=dict)
    total_timesteps: int = 1_000_000  # 튜닝용 짧은 학습


# ──────────────────────────────────────────────
# 실험 세트 정의
# ──────────────────────────────────────────────

REWARD_WEIGHT_EXPERIMENTS = [
    Experiment(
        name="high_forward",
        description="전진 보상 2배 (빠른 보행 유도)",
        config_overrides={"FORWARD_REWARD_WEIGHT": 2.5},
    ),
    Experiment(
        name="low_ctrl_cost",
        description="제어 비용 감소 (자유로운 움직임 허용)",
        config_overrides={"CTRL_COST_WEIGHT": 0.01},
    ),
    Experiment(
        name="low_healthy_reward",
        description="생존 보상 감소 (서있기만 하지 않도록)",
        config_overrides={"HEALTHY_REWARD": 1.0},
    ),
    Experiment(
        name="balanced",
        description="높은 전진 + 낮은 생존 (순수 보행 보상)",
        config_overrides={"FORWARD_REWARD_WEIGHT": 2.5, "HEALTHY_REWARD": 2.0},
    ),
]

LEARNING_RATE_EXPERIMENTS = [
    Experiment(name="lr_1e4", description="LR=1e-4 (높은 학습률)", config_overrides={"LEARNING_RATE": 1e-4}),
    Experiment(name="lr_3e5", description="LR=3.57e-5 (기본값)", config_overrides={}),
    Experiment(name="lr_1e5", description="LR=1e-5 (낮은 학습률)", config_overrides={"LEARNING_RATE": 1e-5}),
]

BATCH_SIZE_EXPERIMENTS = [
    Experiment(name="batch_64", description="Batch=64 (작은 배치)", config_overrides={"BATCH_SIZE": 64}),
    Experiment(name="batch_256", description="Batch=256 (기본값)", config_overrides={}),
    Experiment(name="batch_512", description="Batch=512 (큰 배치)", config_overrides={"BATCH_SIZE": 512}),
]

EXPERIMENT_SETS = {
    "reward_weights": REWARD_WEIGHT_EXPERIMENTS,
    "learning_rates": LEARNING_RATE_EXPERIMENTS,
    "batch_sizes": BATCH_SIZE_EXPERIMENTS,
}


def get_config_value(key: str, overrides: dict):
    """config에서 값을 가져오되, overrides가 있으면 그 값을 사용."""
    return overrides.get(key, getattr(config, key))


def run_experiment(experiment: Experiment) -> dict:
    """단일 실험 실행: 학습 → 평가 → 결과 반환.

    고유한 TensorBoard 로그 디렉토리를 사용하여 비교 가능.
    """
    print(f"\n{'─' * 50}")
    print(f"  Experiment: {experiment.name}")
    print(f"  {experiment.description}")
    print(f"  Overrides: {experiment.config_overrides}")
    print(f"{'─' * 50}")

    exp_log_dir = config.LOG_DIR / f"tune_{experiment.name}"
    exp_model_dir = config.MODEL_DIR / f"tune_{experiment.name}"
    exp_log_dir.mkdir(parents=True, exist_ok=True)
    exp_model_dir.mkdir(parents=True, exist_ok=True)

    overrides = experiment.config_overrides

    # 환경 생성 (override된 reward weights 적용)
    env = DummyVecEnv([lambda: gym.make(
        config.ENV_ID,
        forward_reward_weight=get_config_value("FORWARD_REWARD_WEIGHT", overrides),
        ctrl_cost_weight=get_config_value("CTRL_COST_WEIGHT", overrides),
        contact_cost_weight=get_config_value("CONTACT_COST_WEIGHT", overrides),
        healthy_reward=get_config_value("HEALTHY_REWARD", overrides),
        healthy_z_range=config.HEALTHY_Z_RANGE,
    )])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 모델 생성
    policy_kwargs = dict(
        net_arch=config.NET_ARCH,
        activation_fn=nn.ReLU,
        log_std_init=config.LOG_STD_INIT,
        ortho_init=config.ORTHO_INIT,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=get_config_value("LEARNING_RATE", overrides),
        n_steps=config.N_STEPS,
        batch_size=get_config_value("BATCH_SIZE", overrides),
        n_epochs=config.N_EPOCHS,
        gamma=config.GAMMA,
        gae_lambda=config.GAE_LAMBDA,
        clip_range=config.CLIP_RANGE,
        max_grad_norm=config.MAX_GRAD_NORM,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(exp_log_dir),
        verbose=0,
        seed=42,
        device=config.DEVICE,
    )

    # 학습
    model.learn(total_timesteps=experiment.total_timesteps, progress_bar=True)

    # 모델 저장
    model.save(str(exp_model_dir / "model"))
    env.save(str(exp_model_dir / "vec_normalize.pkl"))

    # 간단 평가 (5 에피소드)
    env.training = False
    env.norm_reward = False

    rewards = []
    lengths = []
    for _ in range(5):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            steps += 1
        rewards.append(total_reward)
        lengths.append(steps)

    env.close()

    results = {
        "name": experiment.name,
        "description": experiment.description,
        "mean_reward": np.mean(rewards),
        "mean_length": np.mean(lengths),
    }

    print(f"  Result: reward={results['mean_reward']:.1f}, ep_len={results['mean_length']:.0f}")
    return results


def run_experiment_set(experiments: list[Experiment], set_name: str) -> None:
    """실험 세트 실행 후 비교 테이블 출력."""
    print(f"\n{'=' * 60}")
    print(f"  Experiment Set: {set_name}")
    print(f"  {len(experiments)} experiments")
    print(f"{'=' * 60}")

    all_results = []
    for exp in experiments:
        result = run_experiment(exp)
        all_results.append(result)

    # 비교 테이블 출력
    print(f"\n{'=' * 60}")
    print(f"  Results Comparison: {set_name}")
    print(f"{'=' * 60}")
    print(f"  {'Name':<20} {'Reward':>10} {'Ep Length':>10}  Description")
    print(f"  {'─' * 70}")
    for r in sorted(all_results, key=lambda x: x["mean_reward"], reverse=True):
        print(f"  {r['name']:<20} {r['mean_reward']:>10.1f} {r['mean_length']:>10.0f}  {r['description']}")

    print(f"\n  TensorBoard로 학습 곡선 비교:")
    print(f"  tensorboard --logdir {config.LOG_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning experiments")
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["reward_weights", "learning_rates", "batch_sizes", "all"],
        help="Which experiment set to run",
    )
    args = parser.parse_args()

    if args.experiment == "all":
        for set_name, experiments in EXPERIMENT_SETS.items():
            run_experiment_set(experiments, set_name)
    else:
        run_experiment_set(EXPERIMENT_SETS[args.experiment], args.experiment)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
