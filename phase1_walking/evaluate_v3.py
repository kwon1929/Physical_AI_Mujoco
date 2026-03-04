"""Evaluate trained G1 v3 agent (Lean Walking)."""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo

from phase1_walking import config_v3 as config
from phase1_walking.g1_env_v3 import G1WalkEnvV3  # Import to register


def evaluate(
    model_path: Path,
    vec_normalize_path: Path,
    n_episodes: int = 5,
    render: bool = False,
    record: bool = False,
    deterministic: bool = True,
):
    """평가 실행."""
    print("=" * 60)
    print(" 🎬 G1 v3 평가 (Lean Walking)")
    print("=" * 60)
    print(f"\n📦 모델: {model_path}")
    print(f"📊 에피소드 수: {n_episodes}")
    print(f"🎥 녹화: {'Yes' if record else 'No'}")
    print(f"👁️  렌더링: {'Yes' if render else 'No'}\n")

    # 환경 생성
    if record:
        config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)

        def make_env():
            env = gym.make(
                config.ENV_ID,
                render_mode="rgb_array",
                forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
                ctrl_cost_weight=config.CTRL_COST_WEIGHT,
                healthy_reward=config.HEALTHY_REWARD,
                healthy_z_range=config.HEALTHY_Z_RANGE,
                action_scale=config.ACTION_SCALE,
                frame_skip=config.FRAME_SKIP,
            )
            env = RecordVideo(
                env,
                str(config.VIDEO_DIR),
                name_prefix="g1_v3_eval",
                episode_trigger=lambda ep: True,
            )
            return env

        env = DummyVecEnv([make_env])
    else:
        render_mode = "human" if render else None
        env = DummyVecEnv([
            lambda: gym.make(
                config.ENV_ID,
                render_mode=render_mode,
                forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
                ctrl_cost_weight=config.CTRL_COST_WEIGHT,
                healthy_reward=config.HEALTHY_REWARD,
                healthy_z_range=config.HEALTHY_Z_RANGE,
                action_scale=config.ACTION_SCALE,
                frame_skip=config.FRAME_SKIP,
            )
        ])

    # VecNormalize 로드
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    # 모델 로드
    model = PPO.load(str(model_path), env=env)
    print("✅ 모델 로드 완료\n")

    # 평가
    episode_rewards = []
    episode_lengths = []
    episode_distances = []

    for ep in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            episode_reward += reward[0]
            steps += 1

            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(steps)

                # 최종 정보 추출
                x_pos = info[0].get('x_position', 0)
                total_dist = info[0].get('total_forward_distance', 0)
                pelvis_h = info[0].get('pelvis_height', 0)

                episode_distances.append(total_dist)

                print(f"Episode {ep + 1}:")
                print(f"  Steps: {steps}")
                print(f"  Reward: {episode_reward:.1f}")
                print(f"  Distance: {total_dist:.3f}m")
                print(f"  Final X: {x_pos:.3f}m")
                print(f"  Final Pelvis Z: {pelvis_h:.3f}m\n")
                break

    # 통계 출력
    print("=" * 60)
    print(" 📊 평가 결과")
    print("=" * 60)
    print(f"\nEpisode Rewards: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Episode Lengths: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Forward Distance: {np.mean(episode_distances):.3f}m ± {np.std(episode_distances):.3f}m")

    # 성공 평가
    avg_length = np.mean(episode_lengths)
    avg_distance = np.mean(episode_distances)

    print("\n" + "=" * 60)
    if avg_length >= 200 and avg_distance >= 3.0:
        print(" ✅ SUCCESS: 로봇이 안정적으로 걷고 있습니다!")
    elif avg_length >= 100 and avg_distance >= 1.0:
        print(" ⚠️  PARTIAL: 로봇이 걷지만 더 개선 가능합니다.")
    elif avg_length >= 50:
        print(" 🤔 PROGRESS: 로봇이 서있지만 충분히 걷지 않습니다.")
    else:
        print(" ❌ FAILED: 로봇이 빨리 넘어집니다.")
    print("=" * 60)

    if record:
        print(f"\n🎥 비디오 저장: {config.VIDEO_DIR}/")
        print(f"   Finder에서 열기: open {config.VIDEO_DIR}\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate G1 v3 agent")
    parser.add_argument(
        "--model",
        type=str,
        default="models/ppo_g1_v3/best_model",
        help="Model path (without .zip extension)",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default="models/ppo_g1_v3/vec_normalize.pkl",
        help="VecNormalize path",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (MuJoCo viewer - may not work on macOS)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record video to videos/ directory",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (default: deterministic)",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    vec_normalize_path = Path(args.vec_normalize)

    if not model_path.exists() and not model_path.with_suffix(".zip").exists():
        print(f"❌ Error: Model not found at {model_path}")
        print(f"\n💡 Hint: Train the model first:")
        print(f"   python -m phase1_walking.train_v3\n")
        return

    if not vec_normalize_path.exists():
        print(f"❌ Error: VecNormalize not found at {vec_normalize_path}")
        return

    evaluate(
        model_path=model_path,
        vec_normalize_path=vec_normalize_path,
        n_episodes=args.n_episodes,
        render=args.render,
        record=args.record,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
