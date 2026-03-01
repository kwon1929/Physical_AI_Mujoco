"""Week 2-3: Evaluate trained agent and record video.

Run: python -m phase1_walking.evaluate
     python -m phase1_walking.evaluate --model-path models/ppo_humanoid/best_model
     python -m phase1_walking.evaluate --record
     python -m phase1_walking.evaluate --render
"""
import argparse

import gymnasium as gym
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecVideoRecorder

from phase1_walking import config


def load_model_and_env(
    model_path: Path,
    vec_normalize_path: Path,
    render_mode: str | None = None,
) -> tuple[PPO, VecNormalize]:
    """학습된 모델과 VecNormalize 통계를 로드.

    중요: VecNormalize stats를 반드시 모델과 함께 로드해야 함.
    정책은 정규화된 관측값으로 학습되었기 때문에, stats 없이 로드하면
    에이전트가 랜덤하게 행동함.
    """
    # 환경 생성
    env = DummyVecEnv([
        lambda: gym.make(config.ENV_ID, render_mode=render_mode)
    ])

    # VecNormalize 로드 (학습 시 저장한 running stats)
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False     # running stats 업데이트 안 함
    env.norm_reward = False  # 원래 reward 값 사용

    # 모델 로드
    model = PPO.load(str(model_path), env=env, device=config.DEVICE)

    return model, env


def run_evaluation(model: PPO, env: VecNormalize, n_episodes: int = 10) -> dict:
    """결정적 평가 에피소드 실행.

    Returns:
        dict: mean_reward, std_reward, mean_length, std_length,
              min_length, max_length, success_rate (100 step 이상 생존)
    """
    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep + 1:>3}: reward={total_reward:>8.2f}, length={steps:>5}")

    results = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "success_rate": sum(1 for l in lengths if l >= 100) / len(lengths),
    }
    return results


def record_video(
    model_path: Path,
    vec_normalize_path: Path,
    video_dir: Path,
    video_length: int = 1000,
    name_prefix: str = "humanoid_walking",
) -> Path:
    """학습된 에이전트의 비디오를 녹화.

    Returns:
        저장된 비디오 파일 경로
    """
    video_dir.mkdir(parents=True, exist_ok=True)

    # rgb_array 모드로 환경 생성
    env = DummyVecEnv([
        lambda: gym.make(config.ENV_ID, render_mode="rgb_array")
    ])
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    # 비디오 레코더 래핑
    env = VecVideoRecorder(
        env,
        str(video_dir),
        record_video_trigger=lambda x: x == 0,  # 첫 에피소드만 녹화
        video_length=video_length,
        name_prefix=name_prefix,
    )

    # 모델 로드 + 실행
    model = PPO.load(str(model_path), env=env, device=config.DEVICE)
    obs = env.reset()

    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done[0]:
            obs = env.reset()

    env.close()

    print(f"  비디오 저장 완료: {video_dir}/")
    return video_dir


def render_live(model_path: Path, vec_normalize_path: Path) -> None:
    """학습된 에이전트를 MuJoCo 뷰어로 실시간 렌더링."""
    print("  MuJoCo 뷰어를 엽니다. 닫으려면 창을 닫거나 Ctrl+C를 누르세요.")

    model, env = load_model_and_env(model_path, vec_normalize_path, render_mode="human")
    obs = env.reset()

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done[0]:
                obs = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("  렌더링 종료.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained humanoid agent")
    parser.add_argument(
        "--model-path", type=str,
        default=str(config.MODEL_DIR / "best_model"),
        help="Path to saved model (without .zip extension)",
    )
    parser.add_argument("--record", action="store_true", help="Record evaluation video")
    parser.add_argument("--render", action="store_true", help="Show live rendering window")
    parser.add_argument("--n-episodes", type=int, default=10, help="Number of evaluation episodes")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    vec_normalize_path = config.MODEL_DIR / "vec_normalize.pkl"

    # 파일 존재 확인
    zip_path = model_path.with_suffix(".zip") if not model_path.suffix else model_path
    if not zip_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {zip_path}")
        print(f"먼저 학습을 실행하세요: python -m phase1_walking.train")
        return

    if not vec_normalize_path.exists():
        print(f"VecNormalize 파일을 찾을 수 없습니다: {vec_normalize_path}")
        return

    print("=" * 60)
    print(" Phase 1: Humanoid-v5 Evaluation")
    print(f" Model: {model_path}")
    print("=" * 60)

    if args.record:
        print("\n--- Recording Video ---")
        record_video(model_path, vec_normalize_path, config.VIDEO_DIR)

    if args.render:
        print("\n--- Live Rendering ---")
        render_live(model_path, vec_normalize_path)

    if not args.record and not args.render:
        # 기본: 평가만 실행
        print(f"\n--- Evaluation ({args.n_episodes} episodes) ---")
        model, env = load_model_and_env(model_path, vec_normalize_path)
        results = run_evaluation(model, env, n_episodes=args.n_episodes)
        env.close()

        print(f"\n--- Results ---")
        print(f"  Mean reward:  {results['mean_reward']:>8.2f} (+/- {results['std_reward']:.2f})")
        print(f"  Mean length:  {results['mean_length']:>8.1f} (+/- {results['std_length']:.1f})")
        print(f"  Min/Max len:  {results['min_length']} / {results['max_length']}")
        print(f"  Success rate: {results['success_rate'] * 100:.0f}% (episodes >= 100 steps)")

        # Phase 1 성공 기준 체크
        if results["mean_length"] >= 100:
            print(f"\n  [PASS] Phase 1 성공 기준 달성! (mean_length >= 100)")
        else:
            print(f"\n  [FAIL] Phase 1 성공 기준 미달 (mean_length < 100)")
            print(f"         더 많은 timestep으로 학습하거나 하이퍼파라미터를 조정하세요.")


if __name__ == "__main__":
    main()
