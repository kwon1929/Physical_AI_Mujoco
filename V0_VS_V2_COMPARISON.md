# G1 Walking: v0 vs v2 환경 비교

## 문제 진단 (v0)

**증상:**
- 2M timesteps 학습 후에도 로봇이 걷지 않음
- 평균 에피소드 길이: 100-120 steps
- 평균 전진 거리: 0.77m (거의 제자리)
- 높은 reward (2000+) 받지만 실제로는 서 있기만 함

**근본 원인:**
1. **Unbounded Linear Reward**: `forward_reward = 25.0 × x_velocity`
   - 작은 움직임으로도 큰 reward 폭발 가능
   - Value function 학습 불안정

2. **부족한 정보**:
   - 이전 액션 정보 없음 (temporal blindness)
   - 발 접촉 정보 없음 (gait pattern 학습 불가)

3. **탐색 부족**:
   - ENT_COEF = 0.01 (너무 낮음)
   - LOG_STD_INIT = -1.0 (초기 분산 작음)
   - ACTION_SCALE = 0.3 (움직임 제한적)

---

## v2 환경 개선사항

### 1. 🎯 Tanh Normalized Forward Reward

**Before (v0):**
```python
# Unbounded: -∞ ~ +∞
forward_reward = 25.0 * x_velocity
```

**After (v2):**
```python
# Bounded: -2.0 ~ +2.0
forward_reward = 2.0 * np.tanh(x_velocity)
```

**효과:**
- Reward explosion 방지
- 안정적인 value function 학습
- Weight를 낮춰도 됨 (25.0 → 2.0)

---

### 2. 🔄 Action Rate Penalty (NEW!)

```python
action_diff = action - self._prev_action
action_rate_cost = 0.1 * np.sum(action_diff ** 2)
```

**효과:**
- 급격한 동작 변화 억제
- 부드러운 걸음걸이 유도
- 29 DoF의 협응 운동 학습

---

### 3. 👣 Foot Contact Rewards (NEW!)

```python
# 발 접촉 감지
foot_contacts = self._get_foot_contacts()  # [left, right]

# 안정성: 최소 한 발은 땅에
at_least_one_contact = np.any(foot_contacts > 0.5)
contact_stability = 0.5 if at_least_one_contact else -0.5

# 교대로 걷기: 한 발씩 번갈아 딛기
exclusive_contact = np.sum(foot_contacts) == 1.0
contact_consistency = 0.3 if exclusive_contact else 0.0
```

**효과:**
- 양발 교대 gait pattern 유도
- 균형 유지 장려
- 현실적인 걸음걸이

---

### 4. ⚡ Energy Efficiency (NEW!)

```python
energy_cost = 0.001 * np.sum(np.abs(torques × velocities))
```

**효과:**
- 비효율적인 움직임 억제
- 에너지 낭비 방지
- 부드럽고 효율적인 보행

---

### 5. 🧠 Temporal Information (NEW!)

**Observation Space:**
- Before (v0): 69 dims (qpos[2:] + qvel)
- After (v2): 98 dims (qpos[2:] + qvel + **prev_action**)

**효과:**
- 시간적 맥락 정보 제공
- Action rate penalty와 시너지
- 더 스마트한 정책 학습

---

### 6. 🚀 Increased Exploration

| Parameter | v0 | v2 | 이유 |
|-----------|----|----|------|
| ENT_COEF | 0.01 | **0.03** | 더 많은 탐색 |
| LOG_STD_INIT | -1.0 | **-0.5** | 초기 분산 증가 |
| ACTION_SCALE | 0.3 | **0.5** | 더 큰 움직임 허용 |

---

## Reward Function 비교

### v0 환경
```python
reward = (
    forward_reward          # 25.0 × velocity (Unbounded!)
    + healthy_reward        # 0.5 (서있기만 해도 OK)
    - ctrl_cost             # 0.1 × action²
)
```

**문제점:**
- Forward reward 폭발 가능
- Healthy reward로 서있기가 안전
- 걷기를 유도할 명시적 신호 없음

---

### v2 환경
```python
reward = (
    forward_reward              # 2.0 × tanh(velocity) ✅ Bounded
    + healthy_reward            # 0.2 (낮춤)
    + contact_stability         # 0.5 (발 접촉)
    + contact_consistency       # 0.3 (교대로 걷기)
    - ctrl_cost                 # 0.01 × action²
    - action_rate_cost          # 0.1 × (Δaction)² ✅ NEW
    - energy_cost               # 0.001 × |τ·ω| ✅ NEW
)
```

**개선점:**
- 모든 reward가 비슷한 스케일 (0.1~2.0 범위)
- 명시적인 gait pattern 유도
- Smoothness와 efficiency 장려

---

## 사용 방법

### v0 환경 (기존)
```bash
# 학습
python -m phase1_walking.train

# 평가
python -m phase1_walking.evaluate --record
```

### v2 환경 (개선)
```bash
# 학습
python -m phase1_walking.train_v2

# 평가
python -m phase1_walking.evaluate_v2 --record

# TensorBoard
tensorboard --logdir logs/ppo_g1_v2
```

---

## 예상 결과

### v0 성능
- ❌ Episode length: ~108 steps
- ❌ Forward distance: ~0.77m
- ❌ Behavior: 제자리에서 균형만 유지

### v2 목표
- ✅ Episode length: >= 200 steps
- ✅ Forward distance: >= 3.0m
- ✅ Behavior: 안정적이고 부드러운 걸음걸이

---

## 추가 자료

- **Reward Shaping 전략**: `REWARD_SHAPING_GUIDE.md`
- **환경 코드**: `phase1_walking/g1_env_v2.py`
- **설정**: `phase1_walking/config_v2.py`

---

## 다음 단계

1. **v2 학습 실행**:
   ```bash
   python -m phase1_walking.train_v2
   ```

2. **TensorBoard 모니터링**:
   - Episode length 증가 확인
   - Forward distance 증가 확인
   - Mean reward 안정성 확인

3. **평가 및 비디오**:
   ```bash
   python -m phase1_walking.evaluate_v2 --record
   open videos/
   ```

4. **성공 기준**:
   - [ ] Episode length >= 200
   - [ ] Forward distance >= 3.0m
   - [ ] 시각적으로 걷는 모습 확인

---

**🎉 v2로 드디어 G1이 걷기를 배울 수 있을 것입니다!**
