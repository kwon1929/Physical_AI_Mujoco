"""Unitree G1 환경 v5 - Linear Forward + Stability Termination

V3 + V4 장점 결합:
- V3에서: 선형 forward reward (서있으면 0, 걸어야 보상)
- V4에서: Roll/Pitch termination (누워서 가기 방지)
- healthy_reward 최소화 (서있기만 하면 손해)
- healthy_z_range 강화 (z >= 0.5m, 누워서 가기 불가)

보상 구조:
  reward = forward_velocity * 10.0  (선형, 걸어야만 보상)
         + healthy_reward * 0.5     (서있기만으론 부족)
         - ctrl_cost * 0.001       (최소 페널티)
  termination: z < 0.5m OR roll/pitch > 0.8 rad
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from phase1_walking.config import MENAGERIE_DIR


class G1WalkEnvV5(gym.Env):
    """Unitree G1 보행 학습 환경 v5 (Linear Forward + Stability Termination)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        # Reward weights (선형! 서있으면 0)
        forward_reward_weight: float = 10.0,
        healthy_reward: float = 0.5,        # 낮게 (서있기만으론 부족)
        ctrl_cost_weight: float = 0.001,    # 최소화
        # Termination (V4에서 가져옴)
        healthy_z_range: tuple[float, float] = (0.5, 1.5),  # 강화! 0.25→0.5
        max_roll_pitch: float = 0.8,        # ~45도, 누워서 가기 방지
        # Other
        max_episode_steps: int = 1000,
        action_scale: float = 0.7,
        frame_skip: int = 5,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.forward_reward_weight = forward_reward_weight
        self.healthy_reward = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_z_range = healthy_z_range
        self.max_roll_pitch = max_roll_pitch
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

        # Foot geom IDs
        try:
            self._left_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link"
            )
            self._right_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link"
            )
        except Exception:
            self._left_foot_geom_id = self.model.ngeom - 2
            self._right_foot_geom_id = self.model.ngeom - 1

        # obs_size 계산
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
        """관측값: qpos (x,y 제외) + qvel + foot contacts."""
        qpos = self.data.qpos[2:].copy()  # z부터 (34 dims)
        qvel = self.data.qvel.copy()       # 전체 속도 (35 dims)
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

    def _get_euler_from_quat(self) -> tuple[float, float, float]:
        """쿼터니언 → 오일러 각도 (roll, pitch, yaw) 변환."""
        w, x, y, z = self.data.qpos[3:7]
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @property
    def pelvis_height(self) -> float:
        return self.data.qpos[2]

    @property
    def roll(self) -> float:
        return self._get_euler_from_quat()[0]

    @property
    def pitch(self) -> float:
        return self._get_euler_from_quat()[1]

    @property
    def is_healthy(self) -> bool:
        """높이 + Roll/Pitch 체크."""
        z = self.pelvis_height
        roll = abs(self.roll)
        pitch = abs(self.pitch)
        height_ok = self.healthy_z_range[0] < z < self.healthy_z_range[1]
        orientation_ok = (roll < self.max_roll_pitch) and (pitch < self.max_roll_pitch)
        return height_ok and orientation_ok

    def step(self, action: np.ndarray):
        x_before = self.data.qpos[0]

        # 액션 적용
        self.data.ctrl[:] = self._default_ctrl + action * self.action_scale

        # 시뮬레이션 스텝
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        x_after = self.data.qpos[0]
        self._step_count += 1

        obs = self._get_obs()

        # ===== 보상 (단순! 선형!) =====
        x_velocity = (x_after - x_before) / self.dt
        forward_reward = self.forward_reward_weight * x_velocity  # 선형
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        reward = forward_reward + healthy_reward - ctrl_cost

        # 종료 조건
        terminated = not self.is_healthy
        truncated = self._step_count >= self._max_episode_steps

        # 정보
        forward_distance = max(0, x_after - x_before)
        self._total_forward_distance += forward_distance

        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "pelvis_height": self.pelvis_height,
            "roll": self.roll,
            "pitch": self.pitch,
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

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)

        noise_scale = 0.01
        self.data.qpos[7:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nq - 7
        )
        self.data.qvel[:] += self.np_random.uniform(
            -noise_scale, noise_scale, size=self.model.nv
        )

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._total_forward_distance = 0.0

        obs = self._get_obs()
        info = {
            "x_position": self.data.qpos[0],
            "pelvis_height": self.pelvis_height,
            "roll": self.roll,
            "pitch": self.pitch,
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
    id="G1Walk-v5",
    entry_point="phase1_walking.g1_env_v5:G1WalkEnvV5",
    max_episode_steps=1000,
)
