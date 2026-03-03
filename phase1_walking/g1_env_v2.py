"""Unitree G1 개선 환경 v2 - Action Rate Penalty + Foot Contact

개선사항:
1. Action Rate Penalty: 급격한 동작 변화 억제
2. Foot Contact Reward: 교대로 발 딛기 유도
3. Forward Progress Tracking: 실제 전진 거리 추적
4. Smooth Reward Shaping: Reward explosion 방지
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

from phase1_walking.config import MENAGERIE_DIR


class G1WalkEnvV2(gym.Env):
    """Unitree G1 보행 학습 환경 v2 (개선된 보상 함수)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: str | None = None,
        # Forward reward - Normalized
        forward_reward_weight: float = 1.0,  # 정규화된 버전
        # Penalties
        ctrl_cost_weight: float = 0.01,
        action_rate_weight: float = 0.1,  # NEW: 액션 변화율 페널티
        energy_weight: float = 0.001,  # NEW: 에너지 소비 페널티
        # Contact rewards
        foot_contact_reward: float = 0.5,  # NEW: 발 접촉 보상
        contact_consistency_weight: float = 0.2,  # NEW: 교대로 걷기 유도
        # Health
        healthy_reward: float = 0.2,  # 낮춤
        healthy_z_range: tuple[float, float] = (0.3, 1.2),
        # Other
        max_episode_steps: int = 1000,
        action_scale: float = 0.5,  # 0.3 → 0.5로 증가
        frame_skip: int = 5,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.action_rate_weight = action_rate_weight
        self.energy_weight = energy_weight
        self.foot_contact_reward = foot_contact_reward
        self.contact_consistency_weight = contact_consistency_weight
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

        # 상태 추적
        self._step_count = 0
        self._prev_action = np.zeros(self.nu)
        self._total_forward_distance = 0.0
        self._prev_x = 0.0

        # Foot contact tracking
        self._prev_foot_contacts = np.array([0.0, 0.0])  # [left, right]

        # Foot geom IDs (G1 모델에서 발 geom 찾기 - 실제 이름 확인 필요)
        try:
            self._left_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_ankle_roll_link"
            )
            self._right_foot_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "right_ankle_roll_link"
            )
        except:
            # 대체: 마지막 두 geom 사용
            self._left_foot_geom_id = self.model.ngeom - 2
            self._right_foot_geom_id = self.model.ngeom - 1

    def _get_obs(self) -> np.ndarray:
        """관측값: qpos (x,y 제외) + qvel + prev_action (temporal info)."""
        qpos = self.data.qpos[2:].copy()  # z부터 (34 dims)
        qvel = self.data.qvel.copy()       # 전체 속도 (35 dims)

        # Temporal information: 이전 액션 추가
        prev_action = self._prev_action.copy()  # 29 dims

        # Total: 34 + 35 + 29 = 98 dims
        return np.concatenate([qpos, qvel, prev_action])

    def _get_foot_contacts(self) -> np.ndarray:
        """발바닥 접촉 감지 (0 or 1)."""
        contacts = np.zeros(2, dtype=np.float32)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if either geom is a foot
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

        # ===== 5. 보상 계산 (개선된 버전) =====

        # 5.1 Forward Reward (Normalized & Clipped)
        x_velocity = (x_after - x_before) / self.dt
        # Sigmoid 정규화: -1 ~ +1 범위로 제한
        forward_reward = self.forward_reward_weight * np.tanh(x_velocity)

        # 5.2 Healthy Reward (작게 유지)
        healthy_reward = self.healthy_reward if self.is_healthy else 0.0

        # 5.3 Control Cost (기존)
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        # 5.4 Action Rate Penalty (NEW!)
        # 급격한 액션 변화를 억제 → 부드러운 동작 유도
        action_diff = action - self._prev_action
        action_rate_cost = self.action_rate_weight * np.sum(np.square(action_diff))

        # 5.5 Energy Cost (NEW!)
        # 실제 토크 × 속도로 에너지 소비 계산
        torques = self.data.ctrl.copy()
        velocities = self.data.qvel[6:].copy()  # 관절 속도만
        energy_cost = self.energy_weight * np.sum(np.abs(torques * velocities))

        # 5.6 Foot Contact Reward (NEW!)
        # 교대로 발을 딛는 패턴 유도
        foot_contacts = self._get_foot_contacts()

        # 적어도 한 발은 땅에 닿아야 함 (안정성)
        at_least_one_contact = np.any(foot_contacts > 0.5)
        contact_stability = self.foot_contact_reward if at_least_one_contact else -0.5

        # 교대로 걷기 유도 (한 발씩 번갈아 딛기)
        # XOR 패턴: 한 발만 닿았을 때 보상
        exclusive_contact = np.sum(foot_contacts) == 1.0
        contact_consistency = (
            self.contact_consistency_weight if exclusive_contact else 0.0
        )

        # ===== 6. 총 보상 =====
        reward = (
            forward_reward
            + healthy_reward
            + contact_stability
            + contact_consistency
            - ctrl_cost
            - action_rate_cost
            - energy_cost
        )

        # ===== 7. 종료 조건 =====
        terminated = not self.is_healthy
        truncated = self._step_count >= self._max_episode_steps

        # ===== 8. 정보 =====
        info = {
            "x_position": x_after,
            "x_velocity": x_velocity,
            "pelvis_height": self.pelvis_height,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "action_rate_cost": action_rate_cost,
            "energy_cost": energy_cost,
            "contact_stability": contact_stability,
            "contact_consistency": contact_consistency,
            "foot_contacts": foot_contacts,
            "total_forward_distance": self._total_forward_distance,
        }

        if self.render_mode == "rgb_array":
            info["rgb_array"] = self.render()

        # ===== 9. 상태 업데이트 =====
        self._prev_action = action.copy()
        forward_distance = max(0, x_after - x_before)  # 후진 무시
        self._total_forward_distance += forward_distance
        self._prev_foot_contacts = foot_contacts

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
        self._prev_action = np.zeros(self.nu)
        self._total_forward_distance = 0.0
        self._prev_x = self.data.qpos[0]
        self._prev_foot_contacts = np.zeros(2)

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
    id="G1Walk-v2",
    entry_point="phase1_walking.g1_env_v2:G1WalkEnvV2",
    max_episode_steps=1000,
)
