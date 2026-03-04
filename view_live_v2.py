"""학습된 G1 v2를 MuJoCo 뷰어로 실시간 시각화.

V2 환경 (개선된 reward shaping)용 뷰어.
Run: python view_live_v2.py
"""
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from phase1_walking import config_v2 as config
from phase1_walking.g1_env_v2 import G1WalkEnvV2  # Import to register


def main():
    print("=" * 60)
    print(" 🤖 학습된 G1 v2 로봇 - 실시간 뷰어")
    print("=" * 60)

    # 모델 로드
    model_path = Path("models/ppo_g1_v2/best_model")
    vec_normalize_path = Path("models/ppo_g1_v2/vec_normalize.pkl")

    if not model_path.with_suffix(".zip").exists():
        print("❌ 모델을 찾을 수 없습니다. 먼저 v2 학습을 실행하세요:")
        print("   python -m phase1_walking.train_v2")
        return

    print("\n📦 모델 로드 중...")

    # V2 환경 생성 (모든 파라미터 명시)
    def make_env():
        return gym.make(
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

    env = DummyVecEnv([make_env])
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
    total_distance = 0.0

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

                # V2 환경의 추가 정보 출력
                x_pos = info[0].get('x_position', 0)
                distance = info[0].get('total_forward_distance', 0)
                pelvis_h = info[0].get('pelvis_height', 0)

                print(f"\nEpisode {episode_count}:")
                print(f"  Steps: {step_count}")
                print(f"  Reward: {reward[0]:.0f}")
                print(f"  Distance: {distance:.3f}m")
                print(f"  Final X: {x_pos:.3f}m")
                print(f"  Final Pelvis Z: {pelvis_h:.3f}m")

                obs = env.reset()
                step_count = 0

    print("\n✅ 뷰어 종료")
    env.close()


if __name__ == "__main__":
    main()
