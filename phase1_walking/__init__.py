"""Phase 1: Humanoid Walking - MuJoCo + PPO reinforcement learning."""
import gymnasium as gym

# Register the custom G1 environment
gym.envs.registration.register(
    id="G1Walk-v0",
    entry_point="phase1_walking.g1_env:G1WalkEnv",
    max_episode_steps=1000,
)
