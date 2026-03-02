"""학습된 G1 에이전트를 실시간 뷰어로 시청.

macOS에서 작동하는 실시간 시각화.
Run: python watch_g1.py [--model-path models/ppo_g1/best_model]
     python watch_g1.py --random  # 랜덤 에이전트 (학습 전 비교용)
"""
import argparse
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase1_walking import config


def watch_trained_agent(model_path: Path, vec_normalize_path: Path):
    """학습된 에이전트를 실시간으로 시청."""
    print("=" * 60)
    print(" 🎬 학습된 G1 에이전트 - 실시간 뷰어")
    print("=" * 60)

    # 모델 로드
    print(f"\n📦 모델 로드: {model_path}")
    env = DummyVecEnv([lambda: gym.make(config.ENV_ID, render_mode="human")])
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    print("✅ 모델 로드 완료\n")

    print("🎬 뷰어 실행 중...")
    print("   ESC 또는 창 닫기로 종료\n")

    obs = env.reset()
    episode = 1
    steps = 0
    total_reward = 0

    try:
        while True:
            # 학습된 정책으로 액션 예측
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # 렌더링
            env.render()

            total_reward += reward[0]
            steps += 1

            # 에피소드 종료 시
            if done[0]:
                x_pos = info[0].get('x_position', 0)
                print(f"Episode {episode}: {steps} steps, "
                      f"reward={total_reward:.0f}, x={x_pos:.2f}m")

                obs = env.reset()
                episode += 1
                steps = 0
                total_reward = 0

    except KeyboardInterrupt:
        print("\n✅ 뷰어 종료")
    finally:
        env.close()


def watch_random_agent():
    """랜덤 에이전트를 실시간으로 시청 (학습 전 비교용)."""
    print("=" * 60)
    print(" 🎲 랜덤 에이전트 - 실시간 뷰어")
    print("=" * 60)

    env = gym.make(config.ENV_ID, render_mode="human")
    print("\n🎬 뷰어 실행 중...")
    print("   ESC 또는 창 닫기로 종료\n")

    obs, info = env.reset()
    episode = 1
    steps = 0
    total_reward = 0

    try:
        while True:
            # 랜덤 액션
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 렌더링
            env.render()

            total_reward += reward
            steps += 1

            # 에피소드 종료 시
            if done:
                x_pos = info.get('x_position', 0)
                print(f"Episode {episode}: {steps} steps, "
                      f"reward={total_reward:.0f}, x={x_pos:.2f}m")

                obs, info = env.reset()
                episode += 1
                steps = 0
                total_reward = 0

    except KeyboardInterrupt:
        print("\n✅ 뷰어 종료")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="G1 에이전트 실시간 시청")
    parser.add_argument(
        "--model-path", type=str,
        default=str(config.MODEL_DIR / "best_model"),
        help="학습된 모델 경로"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="랜덤 에이전트 시청 (학습 전 비교용)"
    )
    args = parser.parse_args()

    if args.random:
        watch_random_agent()
    else:
        model_path = Path(args.model_path)
        vec_normalize_path = config.MODEL_DIR / "vec_normalize.pkl"

        # 파일 존재 확인
        zip_path = model_path.with_suffix(".zip") if not model_path.suffix else model_path
        if not zip_path.exists():
            print(f"❌ 모델을 찾을 수 없습니다: {zip_path}")
            print(f"   먼저 학습을 실행하거나 --random 옵션을 사용하세요.")
            return

        if not vec_normalize_path.exists():
            print(f"❌ VecNormalize 파일을 찾을 수 없습니다: {vec_normalize_path}")
            return

        watch_trained_agent(model_path, vec_normalize_path)


if __name__ == "__main__":
    main()
