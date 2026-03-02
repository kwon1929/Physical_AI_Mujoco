"""학습된 G1를 MuJoCo 뷰어로 실시간 시각화.

macOS 뷰어 실행 버전.
Run: python view_live.py
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
    print(" 🤖 학습된 G1 로봇 - 실시간 뷰어")
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

    # MuJoCo 모델 직접 로드
    print("\n🎬 MuJoCo 뷰어 준비 중...")
    xml_path = str(config.MENAGERIE_DIR / "unitree_g1" / "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("\n🚀 뷰어 실행!")
    print("   마우스: 좌클릭 드래그 = 회전, 우클릭 드래그 = 이동, 스크롤 = 줌")
    print("   ESC 또는 창 닫기로 종료\n")

    # 환경 리셋
    obs = env.reset()
    step_count = 0
    episode_count = 0

    # 뷰어 실행 (launch 사용 - macOS에서 작동)
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # 초기 카메라 위치 설정
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15

        while viewer.is_running():
            # 에이전트 액션 예측
            action, _ = model.predict(obs, deterministic=True)

            # Gym 환경에서 스텝
            obs, reward, done, info = env.step(action)

            # MuJoCo 데이터 동기화 (Gym 환경의 상태를 MuJoCo 뷰어에 복사)
            # Gym 환경의 실제 MuJoCo 데이터 가져오기
            gym_env = env.envs[0].unwrapped
            mj_data.qpos[:] = gym_env.data.qpos
            mj_data.qvel[:] = gym_env.data.qvel

            # Forward kinematics
            mujoco.mj_forward(mj_model, mj_data)

            # 뷰어 업데이트
            viewer.sync()

            step_count += 1

            # 에피소드 종료 시 리셋
            if done[0]:
                episode_count += 1
                print(f"Episode {episode_count}: {step_count} steps, reward: {reward[0]:.0f}")
                obs = env.reset()
                step_count = 0

    print("\n✅ 뷰어 종료")
    env.close()


if __name__ == "__main__":
    main()
