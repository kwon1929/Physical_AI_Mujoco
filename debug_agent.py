"""학습된 에이전트의 동작을 디버그.

모델이 실제로 어떤 액션을 예측하는지 확인.
"""
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym

from phase1_walking import config


def main():
    print("=" * 60)
    print(" 🔍 학습된 G1 에이전트 디버그")
    print("=" * 60)

    # 모델 로드
    model_path = Path("models/ppo_g1/best_model")
    vec_normalize_path = Path("models/ppo_g1/vec_normalize.pkl")

    print("\n📦 모델 로드 중...")
    env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
    env = VecNormalize.load(str(vec_normalize_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    print("✅ 모델 로드 완료\n")

    # 3개 에피소드 실행하며 액션 분석
    for ep in range(3):
        print(f"--- Episode {ep + 1} ---")
        obs = env.reset()
        total_reward = 0
        steps = 0

        # 첫 10 step의 액션 출력
        for i in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            steps += 1

            if i < 10:  # 첫 10 스텝만 출력
                print(f"  Step {i+1}: action mean={action[0].mean():.3f}, "
                      f"std={action[0].std():.3f}, "
                      f"min={action[0].min():.3f}, max={action[0].max():.3f}")

            if done[0]:
                break

        x_pos = info[0].get('x_position', 0)
        pelvis_h = info[0].get('pelvis_height', 0)

        print(f"  Total: {steps} steps, reward={total_reward:.0f}")
        print(f"  Final: x={x_pos:.3f}m, pelvis_z={pelvis_h:.3f}m")
        print(f"  Terminated: {done[0]}\n")

    env.close()
    print("✅ 디버그 완료")


if __name__ == "__main__":
    main()
