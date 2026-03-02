"""Unitree G1 커스텀 Gymnasium 환경.

MuJoCo Menagerie의 G1 모델을 사용하여 보행 학습을 위한 RL 환경.
- Position control (29 actuators)
- 전진 보행 보상 + 생존 보상 - 제어 비용
- "stand" 키프레임에서 시작
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from phase1_walking.config import MENAGERIE_DIR


class G1WalkEnv(gym.Env):
    """Unitree G1 보행 학습 환경.

    Observation (obs_dim):
        - qpos: 관절 위치 (freejoint 7 + 29 joints = 36, x,y 제외 → 34)
        - qvel: 관절 속도 (freejoint 6 + 29 = 35)
        - 합계: 69 dims

    Action (29):
        - 29개 액추에이터의 목표 관절 위치 오프셋 (현재 위치 + delta)

    Reward:
        - 전진 속도 (x방향) * forward_reward_weight
        - 생존 보상 (alive bonus)
        - 제어 비용 패널티
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        forward_reward_weight: float = 1.25,
        healthy_reward: float = 2.0,
        ctrl_cost_weight: float = 0.01,
        healthy_z_range: tuple[float, float] = (0.3, 1.2),
        max_episode_steps: int = 1000,
        action_scale: float = 0.3,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_z_range = healthy_z_range
        self._max_episode_steps = max_episode_steps
        self.action_scale = action_scale

        # MuJoCo 모델 로드
        xml_path = str(MENAGERIE_DIR / "unitree_g1" / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # "stand" 키프레임 ID
        self._stand_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )

        # 키프레임의 기본 ctrl 값 저장 (standing pose)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)
        self._default_ctrl = self.data.ctrl.copy()

        # 공간 정의
        self.nu = self.model.nu  # 29 actuators
        obs_size = self._get_obs().shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        # 액션: [-1, 1] 범위 → action_scale 적용하여 기본 자세 대비 오프셋
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # 렌더러
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        """관측값 생성: qpos (x,y 제외) + qvel."""
        # qpos: [x, y, z, qw, qx, qy, qz, joint1, ..., joint29]
        # x, y 위치는 제외 (에이전트가 위치에 무관하게 학습하도록)
        qpos = self.data.qpos[2:].copy()  # z부터 시작 (34 dims)
        qvel = self.data.qvel.copy()       # 전체 속도 (35 dims)
        return np.concatenate([qpos, qvel])

    @property
    def pelvis_height(self) -> float:
        """골반 높이 (z 좌표)."""
        return self.data.qpos[2]

    @property
    def is_healthy(self) -> bool:
        """로봇이 건강한 상태인지 (넘어지지 않았는지)."""
        z = self.pelvis_height
        return self.healthy_z_range[0] < z < self.healthy_z_range[1]

    @property
    def x_velocity(self) -> float:
        """x 방향 속도 (전진 속도)."""
        return self.data.qvel[0]

    def step(self, action: np.ndarray):
        # 이전 x 위치 기록
        x_before = self.data.qpos[0]

        # 액션 적용: 기본 자세 + 스케일된 오프셋
        self.data.ctrl[:] = self._default_ctrl + action * self.action_scale

        # 시뮬레이션 스텝 (여러 substep)
        mujoco.mj_step(self.model, self.data)

        x_after = self.data.qpos[0]
        self._step_count += 1

        # 관측값
        obs = self._get_obs()

        # 보상 계산
        dt = self.model.opt.timestep
        forward_reward = self.forward_reward_weight * (x_after - x_before) / dt
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        reward = forward_reward + healthy_reward - ctrl_cost

        # 종료 조건
        terminated = not self.is_healthy
        truncated = self._step_count >= self._max_episode_steps

        info = {
            "x_position": x_after,
            "x_velocity": (x_after - x_before) / dt,
            "pelvis_height": self.pelvis_height,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
        }

        if self.render_mode == "rgb_array":
            info["rgb_array"] = self.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # "stand" 키프레임으로 리셋
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)

        # 약간의 노이즈 추가 (탐색 유도)
        noise_scale = 0.01
        self.data.qpos[7:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nq - 7
        )
        self.data.qvel[:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nv
        )

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        obs = self._get_obs()
        info = {"x_position": self.data.qpos[0], "pelvis_height": self.pelvis_height}

        return obs, info

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Gymnasium 환경 등록
gym.register(
    id="G1Walk-v0",
    entry_point="phase1_walking.g1_env:G1WalkEnv",
    max_episode_steps=1000,
)
