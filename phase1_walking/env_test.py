"""Week 1: Environment setup verification and exploration.

Run: python -m phase1_walking.env_test
"""
import gymnasium as gym
import numpy as np

from phase1_walking.config import ENV_ID


# Humanoid-v5 관측값 구조 (exclude_current_positions_from_observation=True 기본값)
OBS_COMPONENTS = {
    "qpos (joint positions, excl. x,y)": (0, 22),    # 22 dims
    "qvel (joint velocities)":           (22, 45),    # 23 dims
    "cinert (body inertias)":            (45, 175),   # 130 dims (13 bodies x 10)
    "cvel (CoM velocities)":             (175, 253),  # 78 dims (13 bodies x 6)
    "qfrc_actuator (actuator forces)":   (253, 270),  # 17 dims
    "cfrc_ext (external forces)":        (270, 348),  # 78 dims (13 bodies x 6)
}

# Humanoid-v5 액추에이터 이름 (17개)
ACTUATOR_NAMES = [
    "abdomen_y", "abdomen_z", "abdomen_x",
    "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee",
    "right_shoulder1", "right_shoulder2", "right_elbow",
    "left_shoulder1", "left_shoulder2", "left_elbow",
]


def print_env_info() -> None:
    """환경의 기본 정보 출력: observation/action space, reward 구조."""
    env = gym.make(ENV_ID)
    obs, info = env.reset()

    print("\n--- Environment Info ---")
    print(f"Environment: {ENV_ID}")
    print(f"Observation space: {env.observation_space}")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Dtype: {env.observation_space.dtype}")
    print(f"  Low:  [{env.observation_space.low.min():.2f}, ...]")
    print(f"  High: [{env.observation_space.high.max():.2f}, ...]")
    print(f"Action space: {env.action_space}")
    print(f"  Shape: {env.action_space.shape}")
    print(f"  Low:  {env.action_space.low[0]:.2f}")
    print(f"  High: {env.action_space.high[0]:.2f}")
    print(f"Max episode steps: {env.spec.max_episode_steps}")

    print(f"\n--- Actuators ({len(ACTUATOR_NAMES)}) ---")
    for i, name in enumerate(ACTUATOR_NAMES):
        print(f"  [{i:2d}] {name}")

    env.close()


def inspect_observation_structure() -> None:
    """348차원 관측값을 각 구성 요소별로 분해하여 통계 출력."""
    env = gym.make(ENV_ID)
    obs, _ = env.reset()

    print(f"\n--- Observation Breakdown (total {obs.shape[0]} dims) ---")
    print(f"{'Component':<40} {'Dims':>5} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 87)

    for name, (start, end) in OBS_COMPONENTS.items():
        component = obs[start:end]
        print(
            f"{name:<40} {end - start:>5} "
            f"{component.mean():>10.4f} {component.std():>10.4f} "
            f"{component.min():>10.4f} {component.max():>10.4f}"
        )

    env.close()


def run_random_episodes(n_episodes: int = 5) -> None:
    """랜덤 액션으로 에피소드를 실행하고 통계 수집."""
    env = gym.make(ENV_ID)

    print(f"\n--- Running {n_episodes} Random Episodes ---")
    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        print(f"  Episode {ep + 1}: reward={total_reward:>8.2f}, length={steps:>5}")

    print(f"\n  Average reward: {np.mean(rewards):.2f} (+/- {np.std(rewards):.2f})")
    print(f"  Average length: {np.mean(lengths):.1f} (+/- {np.std(lengths):.1f})")
    print(f"  Min/Max length: {min(lengths)} / {max(lengths)}")

    env.close()


def visualize_humanoid(n_steps: int = 500) -> None:
    """MuJoCo 뷰어로 휴머노이드 시각화 (랜덤 액션)."""
    print(f"\n--- Visualizing Humanoid (random actions, {n_steps} steps) ---")
    print("  MuJoCo 뷰어 창이 열립니다. 닫으려면 창을 닫거나 Ctrl+C를 누르세요.")

    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()

    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("  시각화 완료.")


def main():
    """Week 1 환경 탐색 전체 실행."""
    print("=" * 60)
    print(" Phase 1, Week 1: Humanoid-v5 Environment Exploration")
    print("=" * 60)

    # 1. 환경 기본 정보
    print_env_info()

    # 2. 관측값 구조 분석
    inspect_observation_structure()

    # 3. 랜덤 에피소드 실행
    run_random_episodes(n_episodes=5)

    # 4. 시각화 (마지막에 실행 - 창이 열림)
    visualize_humanoid(n_steps=500)


if __name__ == "__main__":
    main()
