"""V6g-fix 학습 결과 시각화: 평가(10ep) + mp4 + 학습곡선 + 비교 테이블."""
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from pathlib import Path
import imageio
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from phase1_walking import config_v6 as config
from phase1_walking.g1_env_v6 import G1WalkEnvV6  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_ROOT / "videos"
VIDEO_DIR.mkdir(exist_ok=True)


def make_env(render_mode=None):
    return gym.make(
        config.ENV_ID,
        render_mode=render_mode,
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


def load_model():
    model_path = PROJECT_ROOT / "models" / "ppo_g1_v6g_fix" / "final_model"
    vec_path = PROJECT_ROOT / "models" / "ppo_g1_v6g_fix" / "vec_normalize.pkl"

    env = DummyVecEnv([lambda: make_env()])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(str(model_path), env=env)
    return model, env


def evaluate_and_record(model, env, n_episodes=10):
    """10 에피소드 평가 + best episode mp4 저장."""
    print("=" * 60)
    print(" V6g-fix Evaluation (10 episodes)")
    print("=" * 60)

    results = []

    for ep in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1

            if done[0]:
                ep_data = {
                    "reward": total_reward,
                    "steps": step_count,
                    "x_pos": info[0].get("x_position", 0),
                    "height": info[0].get("pelvis_height", 0),
                    "roll": np.degrees(info[0].get("roll", 0)),
                    "pitch": np.degrees(info[0].get("pitch", 0)),
                    "x_vel": info[0].get("x_velocity", 0),
                    "y_pos": info[0].get("y_position", 0),
                }
                results.append(ep_data)
                print(
                    f"  Ep {ep+1:2d}: steps={step_count:4d}, reward={total_reward:7.0f}, "
                    f"x={ep_data['x_pos']:6.2f}m, z={ep_data['height']:.3f}m, "
                    f"roll={ep_data['roll']:5.1f}, pitch={ep_data['pitch']:5.1f}"
                )
                break

    # Summary
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]
    x_positions = [r["x_pos"] for r in results]
    full_eps = sum(1 for s in steps if s >= 1000)

    print(f"\n  Mean Reward: {np.mean(rewards):.0f} +/- {np.std(rewards):.0f}")
    print(f"  Mean Steps:  {np.mean(steps):.0f} +/- {np.std(steps):.0f}")
    print(f"  Mean X pos:  {np.mean(x_positions):.2f}m +/- {np.std(x_positions):.2f}m")
    print(f"  1000-step episodes: {full_eps}/{n_episodes}")

    return results


def record_video(model, n_episodes=3):
    """rgb_array 환경으로 mp4 녹화."""
    print("\n" + "=" * 60)
    print(" Recording V6g-fix Video")
    print("=" * 60)

    vec_path = PROJECT_ROOT / "models" / "ppo_g1_v6g_fix" / "vec_normalize.pkl"
    vid_env = DummyVecEnv([lambda: make_env(render_mode="rgb_array")])
    vid_env = VecNormalize.load(str(vec_path), vid_env)
    vid_env.training = False
    vid_env.norm_reward = False
    model_for_vid = PPO.load(
        str(PROJECT_ROOT / "models" / "ppo_g1_v6g_fix" / "final_model"), env=vid_env
    )

    all_frames = []

    for ep in range(n_episodes):
        obs = vid_env.reset()
        step_count = 0
        total_reward = 0
        ep_frames = []

        while True:
            action, _ = model_for_vid.predict(obs, deterministic=True)
            obs, reward, done, info = vid_env.step(action)
            total_reward += reward[0]
            step_count += 1

            # rgb_array 가져오기
            gym_env = vid_env.envs[0].unwrapped
            frame = gym_env.render()
            if frame is not None:
                ep_frames.append(frame)

            if done[0]:
                print(
                    f"  Ep {ep+1}: {step_count} steps, reward={total_reward:.0f}, "
                    f"x={info[0].get('x_position', 0):.2f}m"
                )
                break

        all_frames.extend(ep_frames)

    vid_env.close()

    # mp4 저장
    video_path = VIDEO_DIR / "v6g_fix_eval.mp4"
    fps = int(1.0 / (vid_env.envs[0].unwrapped.dt)) if all_frames else 10
    print(f"\n  Saving {len(all_frames)} frames at {fps} fps...")
    imageio.mimsave(str(video_path), all_frames, fps=fps)
    print(f"  Saved: {video_path}")


def eval_checkpoints(model_dir_name, checkpoint_prefix, checkpoint_steps):
    """체크포인트별 평가 → (steps_list, rewards, ep_lengths, x_positions)."""
    vec_path = PROJECT_ROOT / "models" / model_dir_name / "vec_normalize.pkl"
    rewards, ep_lengths, x_positions = [], [], []
    valid_steps = []

    for step in checkpoint_steps:
        ckpt_path = PROJECT_ROOT / "models" / model_dir_name / "checkpoints" / f"{checkpoint_prefix}_{step}_steps"
        if not ckpt_path.with_suffix(".zip").exists():
            continue

        env = DummyVecEnv([lambda: make_env()])
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(str(ckpt_path), env=env)

        ep_r, ep_l, ep_x = [], [], []
        for _ in range(5):
            obs = env.reset()
            total_reward = 0
            step_count = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                step_count += 1
                if done[0]:
                    ep_r.append(total_reward)
                    ep_l.append(step_count)
                    ep_x.append(info[0].get("x_position", 0))
                    break

        env.close()
        valid_steps.append(step)
        rewards.append(np.mean(ep_r))
        ep_lengths.append(np.mean(ep_l))
        x_positions.append(np.mean(ep_x))
        print(f"    {checkpoint_prefix}_{step}: reward={np.mean(ep_r):.0f}, "
              f"steps={np.mean(ep_l):.0f}, x={np.mean(ep_x):.2f}m")

    return valid_steps, rewards, ep_lengths, x_positions


def plot_training_curves():
    """체크포인트 평가 기반 학습 곡선 + TB train metrics."""
    print("\n" + "=" * 60)
    print(" Plotting Training Curves (checkpoint evaluation)")
    print("=" * 60)

    colors = {"V5": "#2196F3", "V6f-ext": "#FF9800", "V6g-fix": "#4CAF50"}

    # V5: eval/mean_reward from TB (100 points available)
    print("\n  [V5] Loading eval data from TensorBoard...")
    ea_v5 = EventAccumulator(
        str(PROJECT_ROOT / "logs" / "ppo_g1_v5" / "PPO_1"),
        size_guidance={"scalars": 0},
    )
    ea_v5.Reload()
    v5_rew_events = ea_v5.Scalars("eval/mean_reward")
    v5_len_events = ea_v5.Scalars("eval/mean_ep_length")
    v5_steps_r = [e.step for e in v5_rew_events]
    v5_rewards = [e.value for e in v5_rew_events]
    v5_steps_l = [e.step for e in v5_len_events]
    v5_lengths = [e.value for e in v5_len_events]
    print(f"    V5 eval: {len(v5_steps_r)} points")

    # V6f-ext: checkpoint evaluation
    print("\n  [V6f-ext] Evaluating checkpoints...")
    ckpt_steps_8m = [1_000_000, 2_000_000, 3_000_000, 4_000_000,
                     5_000_000, 6_000_000, 7_000_000, 8_000_000]

    # Check V6f-ext checkpoint naming
    v6f_ckpt_dir = PROJECT_ROOT / "models" / "ppo_g1_v6f_ext" / "checkpoints"
    v6f_prefix = None
    if v6f_ckpt_dir.exists():
        for f in sorted(v6f_ckpt_dir.iterdir()):
            if f.suffix == ".zip":
                # e.g. "v6f_ext_1000000_steps.zip" → prefix = "v6f_ext"
                name = f.stem  # "v6f_ext_1000000_steps"
                parts = name.rsplit("_", 2)  # ['v6f_ext', '1000000', 'steps']
                if len(parts) >= 3:
                    v6f_prefix = parts[0]
                    print(f"    Found prefix: {v6f_prefix}")
                break

    v6f_steps, v6f_rewards, v6f_lengths, v6f_x = [], [], [], []
    if v6f_prefix:
        v6f_steps, v6f_rewards, v6f_lengths, v6f_x = eval_checkpoints(
            "ppo_g1_v6f_ext", v6f_prefix, ckpt_steps_8m
        )

    # V6g-fix: checkpoint evaluation
    print("\n  [V6g-fix] Evaluating checkpoints...")
    v6g_steps, v6g_rewards, v6g_lengths, v6g_x = eval_checkpoints(
        "ppo_g1_v6g_fix", "v6g_fix", ckpt_steps_8m
    )

    # ─── Plot ───
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Episode Reward
    ax = axes[0]
    ax.plot(np.array(v5_steps_r) / 1e6, v5_rewards, color=colors["V5"],
            linewidth=2, label="V5 (4M)", alpha=0.7)
    if v6f_steps:
        ax.plot(np.array(v6f_steps) / 1e6, v6f_rewards, "o-", color=colors["V6f-ext"],
                linewidth=2, label="V6f-ext (8M)", markersize=6)
    ax.plot(np.array(v6g_steps) / 1e6, v6g_rewards, "s-", color=colors["V6g-fix"],
            linewidth=2, label="V6g-fix (8M)", markersize=6)
    ax.set_xlabel("Timesteps (M)", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("Episode Reward", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 2: Episode Length
    ax = axes[1]
    ax.plot(np.array(v5_steps_l) / 1e6, v5_lengths, color=colors["V5"],
            linewidth=2, label="V5 (4M)", alpha=0.7)
    if v6f_steps:
        ax.plot(np.array(v6f_steps) / 1e6, v6f_lengths, "o-", color=colors["V6f-ext"],
                linewidth=2, label="V6f-ext (8M)", markersize=6)
    ax.plot(np.array(v6g_steps) / 1e6, v6g_lengths, "s-", color=colors["V6g-fix"],
            linewidth=2, label="V6g-fix (8M)", markersize=6)
    ax.set_xlabel("Timesteps (M)", fontsize=12)
    ax.set_ylabel("Mean Episode Length", fontsize=12)
    ax.set_title("Episode Length", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 3: Forward Distance (X position)
    ax = axes[2]
    if v6f_steps:
        ax.plot(np.array(v6f_steps) / 1e6, v6f_x, "o-", color=colors["V6f-ext"],
                linewidth=2, label="V6f-ext (8M)", markersize=6)
    ax.plot(np.array(v6g_steps) / 1e6, v6g_x, "s-", color=colors["V6g-fix"],
            linewidth=2, label="V6g-fix (8M)", markersize=6)
    ax.set_xlabel("Timesteps (M)", fontsize=12)
    ax.set_ylabel("Mean X Position (m)", fontsize=12)
    ax.set_title("Forward Distance", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle("G1 Walking: V5 vs V6f-ext vs V6g-fix Learning Curves", fontsize=16, y=1.02)
    plt.tight_layout()

    plot_path = VIDEO_DIR / "v6g_fix_training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {plot_path}")
    plt.close()


def print_comparison_table():
    """V5 → V6f-ext → V6g-fix 비교 테이블."""
    print("\n" + "=" * 70)
    print(" V5 → V6f-ext → V6g-fix Comparison")
    print("=" * 70)

    # V5, V6f-ext는 기존 데이터 (evaluate_v6_data.py 참조)
    # V6g-fix는 현재 평가에서 수집
    print(f"\n{'Metric':<25} {'V5 (4M)':<20} {'V6f-ext (8M)':<20} {'V6g-fix (8M)':<20}")
    print("-" * 85)

    rows = [
        ("Reward Shape", "linear fwd", "exp(target_vel)", "exp + foot contact"),
        ("Action Scale", "0.7", "0.4", "0.4"),
        ("Total Timesteps", "4M", "8M", "8M"),
        ("Reward Components", "fwd+healthy-ctrl", "+upright+height", "+foot contact"),
        ("Lateral Cost", "-", "0.3*vy + 0.2*y", "0 (removed)"),
        ("Foot Contact", "-", "-", "1.0 (XOR)"),
        ("Termination", "z + roll/pitch", "z + roll/pitch", "z + roll/pitch"),
    ]

    for row in rows:
        print(f"  {row[0]:<23} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    print(f"\n{'Eval Metric':<25} {'V5 (4M)':<20} {'V6f-ext (8M)':<20} {'V6g-fix (8M)':<20}")
    print("-" * 85)

    # Note: V5/V6f-ext data from evaluate_v6_data.py hardcoded baselines
    eval_rows = [
        ("Mean Steps", "~1000", "290", "TBD"),
        ("Mean X Position", "~5.6m", "1.86m", "TBD"),
        ("Mean X Velocity", "~0.3m/s", "0.064m/s", "TBD"),
        ("|Roll| (mean)", "-", "2.3 deg", "TBD"),
        ("|Pitch| (mean)", "-", "5.5 deg", "TBD"),
        ("Y Drift (max)", "-", "1.877m", "TBD"),
        ("1000-step eps", "-", "0/10", "TBD"),
    ]

    for row in eval_rows:
        print(f"  {row[0]:<23} {row[1]:<20} {row[2]:<20} {row[3]:<20}")

    print("\n  (V6g-fix results will be filled from evaluation above)")


def evaluate_all_three():
    """V5, V6f-ext, V6g-fix 모두 평가하여 정확한 비교 테이블 생성."""
    print("\n" + "=" * 70)
    print(" Running Full Comparison: V5 vs V6f-ext vs V6g-fix")
    print("=" * 70)

    versions = [
        ("V5", "ppo_g1_v5", "G1Walk-v5"),
        ("V6f-ext", "ppo_g1_v6f_ext", "G1Walk-v6"),
        ("V6g-fix", "ppo_g1_v6g_fix", "G1Walk-v6"),
    ]

    all_results = {}

    for name, model_dir, env_id in versions:
        model_path = PROJECT_ROOT / "models" / model_dir / "final_model"
        vec_path = PROJECT_ROOT / "models" / model_dir / "vec_normalize.pkl"

        if not model_path.with_suffix(".zip").exists():
            print(f"\n  [{name}] Model not found: {model_path}.zip - skipping")
            continue

        print(f"\n  [{name}] Evaluating...")

        # V5는 다른 env를 사용할 수 있음
        if name == "V5":
            try:
                from phase1_walking.g1_env_v5 import G1WalkEnvV5  # noqa: F401
            except ImportError:
                # V5 env가 없으면 V6 env를 V5 config로 사용
                pass

        env = DummyVecEnv([lambda: make_env()])
        env = VecNormalize.load(str(vec_path), env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(str(model_path), env=env)

        ep_rewards, ep_steps, ep_x, ep_heights = [], [], [], []
        ep_rolls, ep_pitches, ep_y = [], [], []

        for ep in range(10):
            obs = env.reset()
            total_reward = 0
            step_count = 0

            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                step_count += 1

                if done[0]:
                    ep_rewards.append(total_reward)
                    ep_steps.append(step_count)
                    ep_x.append(info[0].get("x_position", 0))
                    ep_heights.append(info[0].get("pelvis_height", 0))
                    ep_rolls.append(abs(np.degrees(info[0].get("roll", 0))))
                    ep_pitches.append(abs(np.degrees(info[0].get("pitch", 0))))
                    ep_y.append(abs(info[0].get("y_position", 0)))
                    break

        env.close()

        all_results[name] = {
            "reward": f"{np.mean(ep_rewards):.0f} +/- {np.std(ep_rewards):.0f}",
            "steps": f"{np.mean(ep_steps):.0f} +/- {np.std(ep_steps):.0f}",
            "x_pos": f"{np.mean(ep_x):.2f}m",
            "height": f"{np.mean(ep_heights):.3f}m",
            "roll": f"{np.mean(ep_rolls):.1f} deg",
            "pitch": f"{np.mean(ep_pitches):.1f} deg",
            "y_drift": f"{np.max(ep_y):.3f}m",
            "full_eps": f"{sum(1 for s in ep_steps if s >= 1000)}/10",
            "raw_steps": np.mean(ep_steps),
            "raw_x": np.mean(ep_x),
            "raw_reward": np.mean(ep_rewards),
        }

        print(
            f"    steps={np.mean(ep_steps):.0f}, reward={np.mean(ep_rewards):.0f}, "
            f"x={np.mean(ep_x):.2f}m, 1000-step: {sum(1 for s in ep_steps if s >= 1000)}/10"
        )

    # 비교 테이블 출력
    if all_results:
        print("\n\n" + "=" * 90)
        print(" COMPARISON TABLE: V5 → V6f-ext → V6g-fix")
        print("=" * 90)

        header = f"  {'Metric':<25}"
        for name in ["V5", "V6f-ext", "V6g-fix"]:
            if name in all_results:
                header += f" {name:<22}"
        print(header)
        print("  " + "-" * (25 + 22 * len(all_results)))

        metric_keys = [
            ("Mean Reward", "reward"),
            ("Mean Steps", "steps"),
            ("Mean X Position", "x_pos"),
            ("Mean Height", "height"),
            ("Mean |Roll|", "roll"),
            ("Mean |Pitch|", "pitch"),
            ("Max Y Drift", "y_drift"),
            ("1000-step Episodes", "full_eps"),
        ]

        for label, key in metric_keys:
            row = f"  {label:<25}"
            for name in ["V5", "V6f-ext", "V6g-fix"]:
                if name in all_results:
                    row += f" {all_results[name][key]:<22}"
            print(row)

    return all_results


def main():
    # 1) 평가 (데이터 수집용)
    model, env = load_model()
    results = evaluate_and_record(model, env, n_episodes=10)
    env.close()

    # 2) 비디오 녹화
    record_video(model, n_episodes=3)

    # 3) 학습 곡선 그래프
    plot_training_curves()

    # 4) 3버전 비교 테이블
    all_results = evaluate_all_three()

    print("\n\n" + "=" * 60)
    print(" All Done!")
    print("=" * 60)
    print(f"  Video:  videos/v6g_fix_eval.mp4")
    print(f"  Graph:  videos/v6g_fix_training_curves.png")
    print()


if __name__ == "__main__":
    main()
