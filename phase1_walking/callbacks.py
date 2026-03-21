"""Custom Stable-Baselines3 callbacks for training monitoring."""
import time
import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv


class ProgressCallback(BaseCallback):
    """콘솔에 학습 진행 상황을 읽기 좋게 출력.

    print_freq 스텝마다:
        - 진행률 (현재/전체 timestep)
        - 최근 100 에피소드 평균 보상
        - 최근 100 에피소드 평균 길이
        - 예상 남은 시간
    """

    def __init__(self, print_freq: int = 10_000, total_timesteps: int = 0, verbose: int = 1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.total_timesteps = total_timesteps
        self.start_time: float | None = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq != 0:
            return True

        elapsed = time.time() - self.start_time
        progress = self.num_timesteps / self.total_timesteps if self.total_timesteps > 0 else 0
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        # 남은 시간 추정
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0
        remaining_min = remaining / 60

        # infos_buffer에서 에피소드 통계 가져오기
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            mean_length = sum(ep["l"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
        else:
            mean_reward = 0.0
            mean_length = 0.0

        print(
            f"[{self.num_timesteps:>9,} / {self.total_timesteps:>9,}] "
            f"{progress * 100:5.1f}% | "
            f"reward: {mean_reward:>8.1f} | "
            f"ep_len: {mean_length:>6.0f} | "
            f"fps: {fps:>5.0f} | "
            f"remaining: {remaining_min:>5.1f}min"
        )

        return True


class EvalByDistanceCallback(BaseCallback):
    """Forward distance 기준으로 best model을 저장하는 평가 콜백.

    SB3 기본 EvalCallback은 mean_reward 기준 → 서있기가 best가 되는 문제.
    ep_length도 서있기가 1000 steps로 최대 → 마찬가지.
    x_position (전진 거리)은 서있기=0, 걷기=1m+ → 확실한 차이.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        model_dir: Path,
        log_dir: Path,
        eval_freq: int = 10_000,
        n_eval_episodes: int = 5,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_distance = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        ep_rewards = []
        ep_lengths = []
        ep_distances = []

        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                total_reward += reward[0]
                step_count += 1
                done = dones[0]

            ep_rewards.append(total_reward)
            ep_lengths.append(step_count)
            ep_distances.append(infos[0].get("x_position", 0.0))

        mean_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        mean_length = np.mean(ep_lengths)
        std_length = np.std(ep_lengths)
        mean_distance = np.mean(ep_distances)

        print(f"Eval num_timesteps={self.num_timesteps}, "
              f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_length:.2f} +/- {std_length:.2f} | "
              f"X distance: {mean_distance:.2f}m")

        if mean_distance > self.best_mean_distance:
            self.best_mean_distance = mean_distance
            self.model.save(str(self.model_dir / "best_model"))
            if self.verbose >= 1:
                print(f"  >> New best mean_x_distance: {mean_distance:.2f}m "
                      f"(steps: {mean_length:.0f}, reward: {mean_reward:.1f}) — model saved!")

        return True


def make_eval_callback(
    eval_env: VecEnv,
    log_dir: Path,
    model_dir: Path,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 5,
) -> EvalCallback:
    """EvalCallback 팩토리: best model 저장 + 평가 로그 기록.

    - eval_freq 스텝마다 에이전트 평가
    - 최고 mean reward 모델을 model_dir/best_model에 저장
    - 평가 결과를 log_dir/evaluations에 기록
    """
    return EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir / "evaluations"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
    )
