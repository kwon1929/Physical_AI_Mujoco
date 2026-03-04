"""Unitree G1 환경 v3 - Lean Walking (단순화, 걷기 집중)

V2 실패 원인:
- 과도한 페널티로 총 reward 음수
- Tanh 정규화로 forward reward 너무 작음
- 로봇이 바로 넘어짐

V3 개선사항:
- Forward reward: 선형 (velocity_x * 10.0)
- Healthy reward: 5.0 (높게)
- 모든 페널티 제거 (ctrl_cost만 0.001로 최소화)
- Action scale 증가 (0.3 → 0.7)
- Termination 완화 (z >= 0.25m)
- Observation에 속도 + 발 접촉 포함
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from phase1_walking.config import MENAGERIE_DIR


class G1WalkEnvV3(gym.Env):
    """Unitree G1 보행 학습 환경 v3 (Lean Walking)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        # Reward weights (단순화!)
        forward_reward_weight: float = 10.0,  # 선형, 높게
        ctrl_cost_weight: float = 0.001,  # 최소화
        healthy_reward: float = 5.0,  # 매 스텝 높은 보상
        # Health
        healthy_z_range: tuple[float, float] = (0.25, 1.5),  # 완화
        # Other
        max_episode_steps: int = 1000,
        action_scale: float = 0.7,  # 증가 (0.3 → 0.7)
        frame_skip: int = 5,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward
        self.healthy_z_range = healthy_z_range
        self._max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.frame_skip = frame_skip

        # MuJoCo 모델 로드
        xml_path = str(MENAGERIE_DIR / "unitree_g1" / "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Effective dt
        self.dt = self.model.opt.timestep * self.frame_skip

        # "stand" 키프레임 ID
        self._stand_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "stand"
        )

        # 키프레임의 기본 ctrl 값 저장
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)
        self._default_ctrl = self.data.ctrl.copy()

        # 공간 정의
        self.nu = self.model.nu  # 29 actuators

        # 상태 추적 변수 먼저 초기화
        self._step_count = 0
        self._total_forward_distance = 0.0
        self._prev_x = 0.0

        # Foot geom IDs
        try:
            self._left_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link"
            )
            self._right_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link"
            )
        except:
            self._left_foot_geom_id = self.model.ngeom - 2
            self._right_foot_geom_id = self.model.ngeom - 1

        # 이제 obs_size 계산 가능
        obs_size = self._get_obs().shape[0]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # 렌더러
        self._renderer = None
        self._viewer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self) -> np.ndarray:
        """관측값: qpos (x,y 제외) + qvel (전체) + foot contacts.

        V3에서는 속도 정보와 발 접촉을 명시적으로 포함.
        """
        qpos = self.data.qpos[2:].copy()  # z부터 (34 dims)
        qvel = self.data.qvel.copy()       # 전체 속도 (35 dims) - 중요!

        # 발 접촉 정보 추가
        foot_contacts = self._get_foot_contacts()  # 2 dims

        # Total: 34 + 35 + 2 = 71 dims
        return np.concatenate([qpos, qvel, foot_contacts])

    def _get_foot_contacts(self) -> np.ndarray:
        """발바닥 접촉 감지 (0 or 1)."""
        contacts = np.zeros(2, dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            if geom1 == self._left_foot_geom_id or geom2 == self._left_foot_geom_id:
                contacts[0] = 1.0
            if geom1 == self._right_foot_geom_id or geom2 == self._right_foot_geom_id:
                contacts[1] = 1.0

        return contacts

    @property
    def pelvis_height(self) -> float:
        return self.data.qpos[2]

    @property
    def is_healthy(self) -> bool:
        z = self.pelvis_height
        return self.healthy_z_range[0] < z < self.healthy_z_range[1]

    def step(self, action: np.ndarray):
        # ===== 1. 이전 상태 기록 =====
        x_before = self.data.qpos[0]

        # ===== 2. 액션 적용 =====
        self.data.ctrl[:] = self._default_ctrl + action * self.action_scale

        # ===== 3. 시뮬레이션 스텝 =====
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = self.data.qpos[0]
        self._step_count += 1

        # ===== 4. 관측값 =====
        obs = self._get_obs()

        # ===== 5. 보상 계산 (단순화!) =====

        # 5.1 Forward Reward (선형, 높은 가중치)
        x_velocity = (x_after - x_before) / self.dt
        forward_reward = self.forward_reward_weight * x_velocity  # 선형!

        # 5.2 Healthy Reward (높게 유지)
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0

        # 5.3 Control Cost (최소화)
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        # 5.4 총 보상 (단순!)
        reward = forward_reward + healthy_reward - ctrl_cost

        # ===== 6. 종료 조건 =====
        terminated = not self.is_healthy
        truncated = self._step_count >= self._max_episode_steps

        # ===== 7. 정보 =====
        forward_distance = max(0, x_after - x_before)
        self._total_forward_distance += forward_distance

        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "pelvis_height": self.pelvis_height,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "total_forward_distance": self._total_forward_distance,
            "foot_contacts": self._get_foot_contacts(),
        }

        if self.render_mode == "rgb_array":
            info["rgb_array"] = self.render()

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # "stand" 키프레임으로 리셋
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)

        # 약간의 노이즈 추가
        noise_scale = 0.01
        self.data.qpos[7:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nq - 7
        )
        self.data.qvel[:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nv
        )

        mujoco.mj_forward(self.model, self.data)

        # 상태 초기화
        self._step_count = 0
        self._total_forward_distance = 0.0
        self._prev_x = self.data.qpos[0]

        obs = self._get_obs()
        info = {
            "x_position": self.data.qpos[0],
            "pelvis_height": self.pelvis_height,
        }

        return obs, info

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        elif self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self._viewer.cam.distance = 5.0
                self._viewer.cam.azimuth = 45
                self._viewer.cam.elevation = -15
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# Gymnasium 환경 등록
gym.register(
    id="G1Walk-v3",
    entry_point="phase1_walking.g1_env_v3:G1WalkEnvV3",
    max_episode_steps=1000,
)
