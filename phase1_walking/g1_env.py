"""Unitree G1 Walking Environment - Custom Gymnasium wrapper.

G1 모델의 특성:
  - 29 actuators (position control, kp=500)
  - 35 DoF (6 free joint + 29 revolute)
  - "stand" keyframe으로 초기화
  - 관측값: qpos, qvel, gyro, accelerometer
  - 액션: 29개 관절 목표 위치 (position control)
"""
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path


class G1WalkEnv(gym.Env):
    """Unitree G1 보행 학습 환경.

    Observation (97 dims):
        - qpos[2:] : 관절 위치 (freejoint z + quat 제외한 x,y 제외) → 34 dims
        - qvel     : 관절 속도 → 35 dims
        - gyro(torso) + accel(torso): IMU 센서 → 6 dims
        - gyro(pelvis) + accel(pelvis): IMU 센서 → 6 dims
        - 이전 액션: → 29 dims (total: 34+35+12+29 = 110 ... 아래 계산)

    실제:
        qpos[2:] = nq - 2 = 37 - 2 = 35 dims (z제외한 freejoint 포함)
        → 더 정확히: qpos[7:] = 30 dims (freejoint 7개 제외, 순수 관절만)
        + qpos[2] = z height (1 dim)
        + qpos[3:7] = orientation quaternion (4 dims)
        총 qpos 부분: 1 + 4 + 30 = 35 dims
        qvel: 35 dims
        sensors: 12 dims
        prev_action: 29 dims  (없어도 됨, 일단 제외)
        ─────────
        Total: 82 dims (without prev_action)

    Action (29 dims):
        각 관절의 목표 위치 오프셋. stand 키프레임 기준으로 delta를 더함.
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
    ):
        super().__init__()

        # 모델 로드
        xml_path = Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # "stand" 키프레임 저장
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._stand_qpos = self.model.key_qpos[key_id].copy()
        self._stand_ctrl = self.model.key_ctrl[key_id].copy()

        # Reward 파라미터
        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_z_range = healthy_z_range
        self._max_episode_steps = max_episode_steps

        # 액션 스페이스: 각 관절의 position target offset
        # stand 키프레임 기준 +-0.5 rad 범위
        nu = self.model.nu  # 29
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(nu,), dtype=np.float32,
        )

        # 관측 스페이스
        obs_size = self._get_obs().shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64,
        )

        # 렌더링
        self.render_mode = render_mode
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

        # 내부 상태
        self._step_count = 0

    def _get_obs(self) -> np.ndarray:
        """현재 상태에서 관측값 생성.

        구성:
          - z_height (1): 골반 높이
          - orientation (4): 골반 quaternion
          - joint_pos (30): 관절 각도 (freejoint 제외)  여기서 nq=37, freejoint=7, so 30
          - qvel (35): 전체 속도
          - sensor_data: gyro + accel (12)
        Total: 1 + 4 + 30 + 35 + 12 = 82
        """
        z_height = self.data.qpos[2:3]           # 1 dim
        orientation = self.data.qpos[3:7]         # 4 dims (quaternion)
        joint_pos = self.data.qpos[7:]            # 30 dims
        qvel = self.data.qvel                     # 35 dims
        sensor_data = self.data.sensordata.copy() # 12 dims (4 sensors x 3)

        return np.concatenate([
            z_height,
            orientation,
            joint_pos,
            qvel,
            sensor_data,
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # stand 키프레임으로 초기화
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._stand_qpos
        self.data.ctrl[:] = self._stand_ctrl

        # 약간의 랜덤 노이즈 추가 (다양한 초기 상태)
        noise = self.np_random.uniform(-0.005, 0.005, size=self.data.qpos.shape)
        noise[:2] = 0  # x, y 위치는 변경하지 않음
        noise[2] = 0   # z 높이도 변경하지 않음
        noise[3:7] = 0  # quaternion도 변경하지 않음
        self.data.qpos += noise

        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._prev_x_pos = self.data.qpos[0]

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        # 이전 x 위치 저장
        x_before = self.data.qpos[0]

        # 액션 적용: stand 키프레임 ctrl + action offset
        self.data.ctrl[:] = self._stand_ctrl + action

        # 시뮬레이션 스텝 (여러 substep)
        n_substeps = 10  # model timestep * n_substeps = control timestep
        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # 관측값
        obs = self._get_obs()

        # 보상 계산
        x_after = self.data.qpos[0]
        forward_velocity = (x_after - x_before) / (self.model.opt.timestep * n_substeps)

        forward_reward = self._forward_reward_weight * forward_velocity
        healthy_reward = self._healthy_reward if self._is_healthy() else 0.0
        ctrl_cost = self._ctrl_cost_weight * np.sum(np.square(action))

        reward = forward_reward + healthy_reward - ctrl_cost

        # 종료 조건
        terminated = not self._is_healthy()
        truncated = self._step_count >= self._max_episode_steps

        info = {
            "forward_velocity": forward_velocity,
            "forward_reward": forward_reward,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "z_height": self.data.qpos[2],
        }

        return obs, reward, terminated, truncated, info

    def _is_healthy(self) -> bool:
        """골반 높이가 범위 내에 있으면 'healthy'."""
        z = self.data.qpos[2]
        return self._healthy_z_range[0] < z < self._healthy_z_range[1]

    def render(self):
        if self.render_mode == "human":
            # human 모드는 별도 뷰어 필요 (view_g1.py 사용)
            pass
        elif self.render_mode == "rgb_array":
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
