"""Unitree G1 환경 v6 - Ablation Reward Engineering

Ablation history:
- V6a: V5 + upright bonus
- V6b: V6a + height bonus
- V6e: forward를 exp(target_vel) + action_scale 축소
- V6f: forward exp 폭 축소 (sigma 1.0→4.0) — 서있기 local optimum 탈출
- V6g: single foot contact + lateral penalty — pitch 안정화 + 직진 유도
- V6h: pitch 계수 2배 + height 조정 + 약한 lateral 재추가

보상 구조 (V6h):
  reward = fw_weight * exp(-fw_sigma * (vel - target_vel)^2)        (목표 속도)
         + upright_weight * exp(-sigma_r * roll^2 - sigma_p * pitch^2) (직립, pitch 강화)
         + height_weight * exp(-sigma_h * (z - target_h)^2)         (높이)
         + single_foot * (left XOR right contact)                    (발 교대)
         + healthy_reward * 0.5
         - ctrl_cost * 0.001
         - lateral_vel_cost * vy^2                                   (횡방향 속도)
         - lateral_pos_cost * y^2                                    (횡방향 표류)
  termination: z < 0.5m OR roll/pitch > 0.8 rad
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from phase1_walking.config import MENAGERIE_DIR


class G1WalkEnvV6(gym.Env):
    """Unitree G1 보행 학습 환경 v6h (pitch 강화 + height 조정 + 약한 lateral)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        # Reward weights
        forward_reward_weight: float = 5.0,    # V6f: exp 기반 forward
        forward_sigma: float = 4.0,            # V6f: exp 폭 (1.0→4.0, 서있기 방지)
        target_velocity: float = 0.5,          # 목표 속도 (m/s)
        upright_reward_weight: float = 3.0,
        upright_sigma: float = 5.0,
        upright_pitch_sigma: float = 10.0,     # V6h: pitch 계수 (roll과 분리)
        height_reward_weight: float = 2.5,     # V6h: 2.0→2.5
        height_sigma: float = 15.0,            # V6h: 10.0→15.0
        target_height: float = 0.75,           # V6h: 0.73→0.75
        healthy_reward: float = 0.5,
        ctrl_cost_weight: float = 0.001,
        # V6g: foot contact + lateral
        single_foot_reward_weight: float = 1.0,
        contact_threshold: float = 16.4,     # G1 mass(33.34) * 9.81 * 0.05
        lateral_vel_cost_weight: float = 0.3,
        lateral_pos_cost_weight: float = 0.2,
        # Termination
        healthy_z_range: tuple[float, float] = (0.5, 1.5),
        max_roll_pitch: float = 0.8,
        # Other
        max_episode_steps: int = 1000,
        action_scale: float = 0.4,            # V6e: 축소 (0.7→0.4)
        frame_skip: int = 5,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.forward_reward_weight = forward_reward_weight
        self.forward_sigma = forward_sigma
        self.target_velocity = target_velocity
        self.upright_reward_weight = upright_reward_weight
        self.upright_sigma = upright_sigma
        self.upright_pitch_sigma = upright_pitch_sigma
        self.height_reward_weight = height_reward_weight
        self.height_sigma = height_sigma
        self.target_height = target_height
        self.healthy_reward = healthy_reward
        self.ctrl_cost_weight = ctrl_cost_weight
        self.single_foot_reward_weight = single_foot_reward_weight
        self.contact_threshold = contact_threshold
        self.lateral_vel_cost_weight = lateral_vel_cost_weight
        self.lateral_pos_cost_weight = lateral_pos_cost_weight
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

        # 상태 추적 변수
        self._step_count = 0
        self._total_forward_distance = 0.0

        # Foot geom sets (body 기반: ankle_roll_link body에 속한 모든 geom)
        left_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
        )
        right_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
        )
        self._left_foot_geoms = set(
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == left_body_id
        )
        self._right_foot_geoms = set(
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == right_body_id
        )

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
        """발바닥 접촉 감지 (0 or 1). Body 기반 geom set 사용."""
        contacts = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            if geoms & self._left_foot_geoms:
                contacts[0] = 1.0
            if geoms & self._right_foot_geoms:
                contacts[1] = 1.0
        return contacts

    def _get_foot_forces(self) -> tuple[float, float]:
        """발바닥 접촉 수직력 (N). Body 기반 geom set 사용."""
        left_force = 0.0
        right_force = 0.0
        force_buf = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geoms = {contact.geom1, contact.geom2}
            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            normal_force = abs(force_buf[0])
            if geoms & self._left_foot_geoms:
                left_force += normal_force
            if geoms & self._right_foot_geoms:
                right_force += normal_force
        return left_force, right_force

    def _get_euler_from_quat(self) -> tuple[float, float, float]:
        """쿼터니언 -> 오일러 각도 (roll, pitch, yaw) 변환."""
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

        # ===== 보상 =====
        x_velocity = (x_after - x_before) / self.dt

        # 1) Forward reward (V6f: sigma=4.0으로 폭 축소, 서있기 방지)
        forward_reward = self.forward_reward_weight * np.exp(
            -self.forward_sigma * ((x_velocity - self.target_velocity) ** 2)
        )

        # 2) Upright reward (V6h: pitch 계수 분리)
        roll = self.roll
        pitch = self.pitch
        upright_reward = self.upright_reward_weight * np.exp(
            -self.upright_sigma * roll**2 - self.upright_pitch_sigma * pitch**2
        )

        # 3) Height reward (V6b)
        z = self.pelvis_height
        height_reward = self.height_reward_weight * np.exp(
            -self.height_sigma * (z - self.target_height)**2
        )

        # 4) Healthy reward
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0

        # 5) Control cost
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        # 6) Single foot contact reward (V6g: 발 교대 유도)
        left_force, right_force = self._get_foot_forces()
        left_contact = left_force > self.contact_threshold
        right_contact = right_force > self.contact_threshold
        single_foot_reward = self.single_foot_reward_weight * float(left_contact != right_contact)

        # 7) Lateral cost (V6g: 횡방향 표류 방지)
        y_position = self.data.qpos[1]
        y_velocity = self.data.qvel[1]
        lateral_cost = (self.lateral_vel_cost_weight * y_velocity**2
                        + self.lateral_pos_cost_weight * y_position**2)

        reward = (forward_reward + upright_reward + height_reward
                  + single_foot_reward + healthy_reward
                  - ctrl_cost - lateral_cost)

        # 종료 조건
        terminated = not self.is_healthy
        truncated = self._step_count >= self._max_episode_steps

        # 정보
        forward_distance = max(0, x_after - x_before)
        self._total_forward_distance += forward_distance

        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "y_position": y_position,
            "y_velocity": y_velocity,
            "pelvis_height": self.pelvis_height,
            "roll": roll,
            "pitch": pitch,
            "forward_reward": forward_reward,
            "upright_reward": upright_reward,
            "height_reward": height_reward,
            "single_foot_reward": single_foot_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "lateral_cost": lateral_cost,
            "left_foot_force": left_force,
            "right_foot_force": right_force,
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
    id="G1Walk-v6",
    entry_point="phase1_walking.g1_env_v6:G1WalkEnvV6",
    max_episode_steps=1000,
)
