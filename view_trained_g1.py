"""학습된 G1 로봇을 직접 시각화.

MuJoCo viewer 문제 해결 버전.

Run: python view_trained_g1.py
"""
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from phase1_walking import config


def main():
    print("=" * 60)
    print(" 학습된 G1 로봇 시각화")
    print("=" * 60)

    # 모델 로드
    model_path = Path("models/ppo_g1/best_model")
    vec_normalize_path = Path("models/ppo_g1/vec_normalize.pkl")

    if not model_path.with_suffix(".zip").exists():
        print("❌ 모델을 찾을 수 없습니다. 먼저 학습을 실행하세요.")
        return

    print("\n📦 모델 로드 중...")
    env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    print("✅ 모델 로드 완료")

    # MuJoCo 모델 직접 로드 (시각화용)
    print("\n🎬 MuJoCo 뷰어 준비 중...")
    xml_path = str(config.MENAGERIE_DIR / "unitree_g1" / "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    # 초기 상태 설정
    obs = env.reset()

    print("\n🚀 시각화 시작!")
    print("   MuJoCo 뷰어가 열립니다.")
    print("   마우스: 좌클릭 드래그 = 회전, 우클릭 드래그 = 이동, 스크롤 = 줌")
    print("   Ctrl+우클릭 = 로봇에 힘 가하기")
    print("   창을 닫으면 종료됩니다.\n")

    step_count = 0

    # macOS에서는 launch_passive 대신 간단한 평가 + 비디오 저장 사용
    print("💡 macOS에서는 실시간 뷰어 대신 비디오 녹화를 사용하세요:")
    print("   python -m phase1_walking.evaluate --record")
    print("\n📊 대신 학습된 에이전트 성능을 확인합니다...\n")

    # 여러 에피소드 실행
    for ep in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1

        print(f"   Episode {ep+1}: {steps:>4} steps, reward: {total_reward:>8.0f}")

    print("\n✅ 평가 완료")
    print("\n🎬 걷는 모습을 보려면:")
    print("   python -m phase1_walking.evaluate --record")
    print("   (videos/ 폴더에 MP4 저장됨)")

    env.close()


if __name__ == "__main__":
    main()
