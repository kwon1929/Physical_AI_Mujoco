# V6b 병목 수정 — 우선순위별 실행 계획

## 현재 상황 요약
V6b에서 로봇이 "안정적 걷기"가 아니라 "sprint & crash"를 반복한다.
- 10개 에피소드 전부 넘어져서 종료 (max 1000 steps 도달 0회)
- 평균 생존: 124 steps (12.4%)
- Forward reward(10*vel)가 step당 보상의 73% 차지
- PPO가 "빨리 달리다 넘어지기" local optimum에 갇혀 있음

## 수정 우선순위

### Priority 1: Forward reward 교체 + Action scale 축소 (V6e 실험)

이 두 가지를 **동시에** 적용한 새 실험 V6e를 만들어줘.

#### 1-1. Forward reward 변경
**현재 (문제):**
```python
forward_reward = 10.0 * x_velocity  # 선형, 속도 무한정 보상
```

**변경:**
```python
# 목표 속도 0.5 m/s에서 최대 보상, 과속하면 감소
target_vel = 0.5
forward_reward = 5.0 * np.exp(-((x_velocity - target_vel) ** 2))
```

이렇게 하면:
- 정지(0 m/s): 5.0 * exp(-0.25) = 3.89
- 0.5 m/s (목표): 5.0 * 1.0 = 5.0 (최대)
- 1.0 m/s: 5.0 * exp(-0.25) = 3.89
- 2.0 m/s: 5.0 * exp(-2.25) = 0.53
- 과속의 인센티브가 사라짐

#### 1-2. Action scale 축소
**현재:**
```python
action_scale = 0.7
```

**변경:**
```python
action_scale = 0.4
```

격렬한 관절 움직임을 물리적으로 제한해서 부드러운 제어를 유도한다.

#### 나머지 보상은 V6b 그대로 유지:
- upright_reward = 3.0 * exp(-5.0 * (roll² + pitch²))  # 라디안 기준
- height_reward = 2.0 * exp(-10.0 * (z - 0.73)²)
- healthy_reward = 0.5
- ctrl_cost = -0.001 * sum(action²)

#### V6e 예상 step당 보상 비중:
| 항목 | 최대값 | 걸을 때(0.5m/s) | 비중 |
|------|--------|-----------------|------|
| forward | 5.0 | 5.0 | 42% |
| upright | 3.0 | ~2.5 | 21% |
| height | 2.0 | ~1.5 | 13% |
| healthy | 0.5 | 0.5 | 4% |
| **합계** | | **~9.5** | |

Forward 비중이 73% → 42%로 떨어지고, stability(upright+height)가 22% → 34%로 올라간다.

#### 실행 및 기록:
- V6b와 동일한 학습 세팅(step 수, 환경 수 등)으로 학습
- 학습 후 다음 메트릭을 V6b와 비교:
  - 평균 episode steps (생존 시간)
  - 평균 x_velocity
  - 평균 roll, pitch (도 단위)
  - 평균 z height
  - 평균 y drift
  - max_episode_steps(1000) 도달 에피소드 비율
  - episode reward 학습 곡선

### Priority 2: Single foot contact 보상 추가 (V6e 결과 확인 후)

V6e에서 생존 시간이 늘어나지만 여전히 hopping/lunging이 보이면 추가한다.

```python
# MuJoCo에서 왼발/오른발 접촉 감지
# 양발 동시 접촉이면 0, 한 발만 접촉이면 1
left_contact = (left_foot_contact_force > threshold)   # bool
right_contact = (right_foot_contact_force > threshold)  # bool

# XOR: 한 발만 접촉할 때 보상
single_foot_bonus = 1.0 * float(left_contact != right_contact)
```

- contact force threshold는 로봇 무게의 ~10% 정도로 설정 (실험적 조정 필요)
- 왼발/오른발의 geom 이름이나 body 이름을 MuJoCo 모델에서 확인해서 접촉력을 가져와야 함
- 보상 가중치 1.0은 시작점이고, V6e 결과를 보고 조정

### Priority 3: Curriculum warm start (Priority 1-2로 안 되면)

V4에서 학습한 standing policy의 가중치를 V6e의 초기 가중치로 사용한다.

```python
# V4 standing policy 로드
standing_policy = load_model("v4_standing_best.zip")  # 또는 해당 경로

# V6e 환경에서 standing policy 가중치로 초기화
v6e_model = PPO("MlpPolicy", v6e_env, ...)
v6e_model.policy.load_state_dict(standing_policy.policy.state_dict())

# 작은 learning rate로 fine-tuning
v6e_model.learning_rate = 1e-4  # 기존보다 낮게
v6e_model.learn(total_timesteps=...)
```

이렇게 하면 PPO가 "서있는 상태"에서 출발해서 "조금씩 앞으로 가기"를 탐색한다.

## 중요: 하지 말아야 할 것
- V6c(lateral penalty)나 V6d(exp forward)를 원래 ablation 순서대로 하지 말 것. Forward의 선형 구조가 근본 원인이므로 이걸 먼저 고쳐야 함
- 학습 step 수를 늘리는 것으로 해결하려 하지 말 것. 보상 구조가 문제이지 학습량이 문제가 아님
- Gait cycle clock signal이나 reference motion은 아직 추가하지 말 것. 복잡도가 급격히 올라감

## 실행 순서
1. V6e(forward exp + action_scale 0.4) 구현 및 학습
2. 결과 메트릭 기록 및 V6b와 비교 테이블 작성
3. 생존 시간 500+ steps 달성 여부로 다음 단계 결정:
   - 달성 O → 영상 확인 후 hopping이면 Priority 2 진행
   - 달성 X → Priority 3(warm start) 진행