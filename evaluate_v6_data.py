"""V6g-fix 실측 데이터 수집 (V6f-ext baseline 비교용)."""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path

from phase1_walking import config_v6 as config
from phase1_walking.g1_env_v6 import G1WalkEnvV6  # noqa: F401


def main():
    model_path = Path("models/ppo_g1_v6g_fix/final_model")
    vec_normalize_path = Path("models/ppo_g1_v6g_fix/vec_normalize.pkl")

    def make_env():
        return gym.make(
            config.ENV_ID,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            forward_sigma=config.FORWARD_SIGMA,
            target_velocity=config.TARGET_VELOCITY,
            upright_reward_weight=config.UPRIGHT_REWARD_WEIGHT,
            upright_sigma=config.UPRIGHT_SIGMA,
            height_reward_weight=config.HEIGHT_REWARD_WEIGHT,
            height_sigma=config.HEIGHT_SIGMA,
            target_height=config.TARGET_HEIGHT,
            healthy_reward=config.HEALTHY_REWARD,
            ctrl_cost_weight=config.CTRL_COST_WEIGHT,
            single_foot_reward_weight=config.SINGLE_FOOT_REWARD_WEIGHT,
            contact_threshold=config.CONTACT_THRESHOLD,
            lateral_vel_cost_weight=config.LATERAL_VEL_COST_WEIGHT,
            lateral_pos_cost_weight=config.LATERAL_POS_COST_WEIGHT,
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

    n_episodes = 10
    all_data = {
        "rewards": [], "steps": [], "x_pos": [],
        "heights": [], "rolls": [], "pitches": [],
        "velocities": [], "upright_rewards": [],
    }
    # Per-step data for distributions
    step_heights = []
    step_rolls = []
    step_pitches = []
    step_velocities = []
    step_y_positions = []

    print(f"Running {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1

            # Per-step data
            step_heights.append(info[0].get("pelvis_height", 0))
            step_rolls.append(abs(np.degrees(info[0].get("roll", 0))))
            step_pitches.append(abs(np.degrees(info[0].get("pitch", 0))))
            step_velocities.append(info[0].get("x_velocity", 0))
            # y position from underlying env
            gym_env = env.envs[0].unwrapped
            step_y_positions.append(abs(gym_env.data.qpos[1]))

            if done[0]:
                all_data["rewards"].append(total_reward)
                all_data["steps"].append(step_count)
                all_data["x_pos"].append(info[0].get("x_position", 0))
                all_data["heights"].append(info[0].get("pelvis_height", 0))
                all_data["rolls"].append(np.degrees(info[0].get("roll", 0)))
                all_data["pitches"].append(np.degrees(info[0].get("pitch", 0)))
                all_data["velocities"].append(info[0].get("x_velocity", 0))

                print(f"  Ep {ep+1}: steps={step_count}, reward={total_reward:.0f}, "
                      f"x={info[0].get('x_position', 0):.2f}m, "
                      f"z={info[0].get('pelvis_height', 0):.3f}m, "
                      f"roll={np.degrees(info[0].get('roll', 0)):.1f}, "
                      f"pitch={np.degrees(info[0].get('pitch', 0)):.1f}")
                break

    env.close()

    # Summary
    print("\n" + "=" * 60)
    print(" V6g-fix Results (Fixed Foot Contact, No Lateral)")
    print("=" * 60)
    print(f"\n  Episodes: {n_episodes}")
    print(f"  Mean Reward: {np.mean(all_data['rewards']):.0f} +/- {np.std(all_data['rewards']):.0f}")
    print(f"  Mean Steps:  {np.mean(all_data['steps']):.0f} +/- {np.std(all_data['steps']):.0f}")
    print(f"  Mean X pos:  {np.mean(all_data['x_pos']):.2f}m +/- {np.std(all_data['x_pos']):.2f}m")

    print(f"\n  Step-level distributions:")
    print(f"    Height:  mean={np.mean(step_heights):.3f}m, std={np.std(step_heights):.3f}, "
          f"min={np.min(step_heights):.3f}, max={np.max(step_heights):.3f}")
    print(f"    |Roll|:  mean={np.mean(step_rolls):.1f}, std={np.std(step_rolls):.1f}, "
          f"max={np.max(step_rolls):.1f}")
    print(f"    |Pitch|: mean={np.mean(step_pitches):.1f}, std={np.std(step_pitches):.1f}, "
          f"max={np.max(step_pitches):.1f}")
    print(f"    X vel:   mean={np.mean(step_velocities):.3f}, std={np.std(step_velocities):.3f}")
    print(f"    |Y pos|: mean={np.mean(step_y_positions):.3f}, max={np.max(step_y_positions):.3f}")

    # Baselines
    print(f"\n  V6f-ext Baseline (8M, EvalByDistance):")
    print(f"    Steps: 290 | X pos: 1.86m | X vel: 0.64 | |Roll|: 2.3")
    print(f"    |Pitch|: 5.5 (max 45.9) | Y drift: 1.877m | 500+: 0/10")

    # 1000 step 도달 비율
    full_eps = sum(1 for s in all_data["steps"] if s >= 1000)
    print(f"\n  1000-step episodes: {full_eps}/{n_episodes}")


if __name__ == "__main__":
    main()
