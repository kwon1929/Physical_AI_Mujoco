"""학습된 G1 v6를 MuJoCo Interactive Viewer로 실시간 시각화.

V6: V5 + Upright Bonus (Ablation Step 1)
Run: ./venv/bin/mjpython view_live_v6_interactive.py
"""
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import time

from phase1_walking import config_v6 as config
from phase1_walking.g1_env_v6 import G1WalkEnvV6  # noqa: F401 (register env)


def main():
    print("=" * 60)
    print(" G1 v6 - Interactive Viewer")
    print(" (V5 + Upright Bonus)")
    print("=" * 60)

    model_path = Path("models/ppo_g1_v6/best_model")
    vec_normalize_path = Path("models/ppo_g1_v6/vec_normalize.pkl")

    if not model_path.with_suffix(".zip").exists():
        print("Model not found. Run training first:")
        print("   python -m phase1_walking.train_v6")
        return

    print("\nLoading model...")

    def make_env():
        return gym.make(
            config.ENV_ID,
            forward_reward_weight=config.FORWARD_REWARD_WEIGHT,
            upright_reward_weight=config.UPRIGHT_REWARD_WEIGHT,
            upright_sigma=config.UPRIGHT_SIGMA,
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
    print("Model loaded")

    # MuJoCo 뷰어용 모델
    xml_path = str(config.MENAGERIE_DIR / "unitree_g1" / "scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)

    print("\nControls:")
    print("  Mouse drag: rotate/pan")
    print("  Scroll: zoom")
    print("  Close window or Ctrl+C: quit\n")

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.distance = 5.0
        viewer.cam.azimuth = 45
        viewer.cam.elevation = -15

        episode_count = 0

        while viewer.is_running():
            obs = env.reset()
            step_count = 0
            total_reward = 0
            episode_count += 1

            print(f"\n--- Episode {episode_count} ---")

            while viewer.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                step_count += 1

                # MuJoCo 데이터 동기화
                gym_env = env.envs[0].unwrapped
                mj_data.qpos[:] = gym_env.data.qpos
                mj_data.qvel[:] = gym_env.data.qvel
                mujoco.mj_forward(mj_model, mj_data)

                viewer.sync()
                time.sleep(0.01)

                if done[0]:
                    x_pos = info[0].get('x_position', 0)
                    pelvis_h = info[0].get('pelvis_height', 0)
                    vel = info[0].get('x_velocity', 0)
                    roll = info[0].get('roll', 0)
                    pitch = info[0].get('pitch', 0)

                    print(f"  Steps: {step_count} | Reward: {total_reward:.0f} | "
                          f"X: {x_pos:.2f}m | Vel: {vel:.2f}m/s | Z: {pelvis_h:.2f}m | "
                          f"Roll: {np.degrees(roll):.1f} | Pitch: {np.degrees(pitch):.1f}")

                    time.sleep(1)
                    break

    env.close()
    print("\nViewer closed.")


if __name__ == "__main__":
    main()
