"""학습된 G1 v4를 MuJoCo Interactive Viewer로 실시간 시각화.

V4: Exponential Reward + Stability (MiniCheetah 기법)
Run: python view_live_v4_interactive.py
"""
import numpy as np
import mujoco
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import time

from phase1_walking import config_v4 as config
from phase1_walking.g1_env_v4 import G1WalkEnvV4  # Import to register


def main():
    print("=" * 60)
    print(" 🤖 학습된 G1 v4 로봇 - Interactive Viewer")
    print(" (Exponential Reward + Stability)")
    print("=" * 60)

    # 모델 로드
    model_path = Path("models/ppo_g1_v4/best_model")
    vec_normalize_path = Path("models/ppo_g1_v4/vec_normalize.pkl")

    if not model_path.with_suffix(".zip").exists():
        print("❌ 모델을 찾을 수 없습니다. 먼저 v4 학습을 실행하세요:")
        print("   python -m phase1_walking.train_v4")
        return

    print("\n📦 모델 로드 중...")

    # V4 환경 생성
    def make_env():
        return gym.make(
            config.ENV_ID,
            target_velocity=config.TARGET_VELOCITY,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            stability_reward_weight=config.STABILITY_REWARD_WEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            healthy_z_range=config.HEALTHY_Z_RANGE,
            max_roll_pitch=config.MAX_ROLL_PITCH,
            action_scale=config.ACTION_SCALE,
            frame_skip=config.FRAME_SKIP,
        )

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    print("✅ 모델 로드 완료")

    # MuJoCo 모델 직접 로드 (뷰어용)
    print("\n🎬 MuJoCo Interactive Viewer 준비 중...")
    xml_path = str(config.MENAGERIE_DIR / "unitree_g1" / "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("\n🚀 Interactive Viewer 실행!")
    print("   조작법:")
    print("   - 마우스 좌클릭 드래그: 회전")
    print("   - 마우스 우클릭 드래그: 이동")
    print("   - 마우스 스크롤: 줌")
    print("   - Space: 일시정지/재생")
    print("   - Backspace: 리셋")
    print("   - Esc: 종료")
    print("   - Tab: 프레임 단위 진행")
    print("\n   V4 특징: 목표 속도 1.0 m/s, 자세 안정성 제어\n")

    episode_count = 0

    while True:
        try:
            # 환경 리셋
            obs = env.reset()
            step_count = 0
            total_reward = 0
            episode_count += 1

            print(f"\n{'='*60}")
            print(f"Episode {episode_count} 시작 (V4)")
            print(f"{'='*60}")

            # Interactive viewer 실행
            with mujoco.viewer.launch(mj_model, mj_data) as viewer:
                # 초기 카메라 위치 설정
                viewer.cam.distance = 5.0
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -15

                while viewer.is_running():
                    # 에이전트 액션 예측
                    action, _ = model.predict(obs, deterministic=True)

                    # Gym 환경에서 스텝
                    obs, reward, done, info = env.step(action)
                    total_reward += reward[0]

                    # MuJoCo 데이터 동기화
                    gym_env = env.envs[0].unwrapped
                    mj_data.qpos[:] = gym_env.data.qpos
                    mj_data.qvel[:] = gym_env.data.qvel

                    # Forward kinematics
                    mujoco.mj_forward(mj_model, mj_data)

                    step_count += 1

                    # 에피소드 종료 시
                    if done[0]:
                        distance = info[0].get('total_forward_distance', 0)
                        x_pos = info[0].get('x_position', 0)
                        pelvis_h = info[0].get('pelvis_height', 0)
                        roll = info[0].get('roll', 0)
                        pitch = info[0].get('pitch', 0)

                        print(f"\nEpisode {episode_count} 종료:")
                        print(f"  Steps: {step_count}")
                        print(f"  Reward: {total_reward:.0f}")
                        print(f"  Distance: {distance:.3f}m")
                        print(f"  Final X: {x_pos:.3f}m")
                        print(f"  Final Pelvis Z: {pelvis_h:.3f}m")
                        print(f"  Final Roll: {np.degrees(roll):.1f}°")
                        print(f"  Final Pitch: {np.degrees(pitch):.1f}°")

                        # 2초 대기 후 종료
                        time.sleep(2)
                        break

            # 다음 에피소드 시작
            print("\n다음 에피소드를 시작합니다...")
            print("(종료하려면 Ctrl+C 누르세요)")
            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n✅ 사용자가 종료했습니다.")
            break

    env.close()
    print("\n✅ 뷰어 종료")


if __name__ == "__main__":
    main()
