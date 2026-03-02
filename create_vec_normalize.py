"""현재 best_model을 위한 vec_normalize.pkl 생성.

학습 중단 시 vec_normalize.pkl이 없을 때 사용.
몇 개 에피소드를 랜덤으로 실행해서 running stats를 수집.
"""
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path

from phase1_walking import config


def create_vec_normalize(n_episodes: int = 10):
    """VecNormalize를 생성하고 초기화."""
    print("=" * 60)
    print(" VecNormalize 생성")
    print("=" * 60)

    # 환경 생성
    print(f"\n📦 환경 생성: {config.ENV_ID}")
    env = DummyVecEnv([lambda: gym.make(config.ENV_ID)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 몇 에피소드 실행해서 running stats 수집
    print(f"\n🏃 {n_episodes}개 에피소드로 통계 수집 중...")
    obs = env.reset()
    for ep in range(n_episodes):
        done = False
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            steps += 1
            if done[0]:
                obs = env.reset()
                break
        print(f"  Episode {ep+1}/{n_episodes}: {steps} steps")

    # 저장
    save_path = config.MODEL_DIR / "vec_normalize.pkl"
    env.save(str(save_path))
    print(f"\n✅ VecNormalize 저장: {save_path}")

    # 통계 출력
    print(f"\n📊 수집된 통계:")
    print(f"  Obs mean: {env.obs_rms.mean[:5]}...")
    print(f"  Obs var:  {env.obs_rms.var[:5]}...")

    env.close()
    return save_path


if __name__ == "__main__":
    create_vec_normalize()
