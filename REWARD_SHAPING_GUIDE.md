# G1 Walking - Reward Shaping 전략 가이드

## 문제: Forward Reward 25.0으로 인한 Reward Explosion

**현상:**
- Forward reward weight = 25.0
- 짧은 거리만 이동해도 reward가 2000+ 폭발
- VecNormalize가 reward를 정규화하지만 학습 불안정

**근본 원인:**
- Linear scaling (25 × velocity)은 제한이 없음
- Reward 범위가 너무 넓어 value function 학습 어려움

---

## 해결책 1: Tanh/Sigmoid 정규화 (⭐ 추천)

### 원리
```python
# 기존 (Unbounded)
forward_reward = 25.0 * x_velocity  # -∞ ~ +∞

# 개선 (Bounded)
forward_reward = weight * np.tanh(x_velocity)  # -weight ~ +weight
```

### 장점
- Reward 범위가 [-1, 1] 또는 [-weight, weight]로 제한
- 매우 빠른 속도에 대해 포화(saturation) 효과
- Value function이 안정적으로 학습됨

### 구현
```python
def step(self, action):
    x_velocity = (x_after - x_before) / self.dt

    # Option 1: Tanh (부드러운 포화)
    forward_reward = self.forward_reward_weight * np.tanh(x_velocity)

    # Option 2: Sigmoid (0~1 범위)
    forward_reward = self.forward_reward_weight * (
        1 / (1 + np.exp(-x_velocity))
    )

    # Option 3: Clipping
    forward_reward = np.clip(
        self.forward_reward_weight * x_velocity,
        -5.0, 5.0
    )
```

**추천 설정:**
- `forward_reward_weight = 1.0~5.0` (낮춰도 됨!)
- Tanh가 자동으로 스케일 조정

---

## 해결책 2: Sparse + Dense 조합

### 원리
```python
# Dense reward (매 스텝)
dense_reward = 0.1 * x_velocity

# Sparse reward (목표 달성 시)
if x_position > threshold:
    sparse_reward = 10.0
else:
    sparse_reward = 0.0

total_reward = dense_reward + sparse_reward
```

### 장점
- Dense reward가 방향 제시
- Sparse reward가 명확한 목표 제공

### 구현
```python
# 거리 기반 Milestone
milestones = [1.0, 2.0, 5.0, 10.0]  # meters
for milestone in milestones:
    if x_position >= milestone and not reached[milestone]:
        reward += 5.0 * (milestone / 10.0)  # 거리에 비례
        reached[milestone] = True
```

---

## 해결책 3: Relative Progress Reward

### 원리
개인 최고 기록 대비 개선도로 보상

```python
# 에피소드별 최고 거리 추적
self.best_distance = 0.0

def step(self, action):
    current_distance = self._total_forward_distance

    if current_distance > self.best_distance:
        progress_reward = 1.0  # 신기록 달성!
        self.best_distance = current_distance
    else:
        progress_reward = 0.0

    reward = progress_reward + other_rewards
```

### 장점
- 절대 속도가 아닌 **개선**에 집중
- Curriculum learning 효과

---

## 해결책 4: Multi-Objective Reward

### 원리
여러 목표를 균형있게 조합

```python
rewards = {
    'forward': np.tanh(x_velocity),           # -1 ~ 1
    'stability': -np.abs(angular_velocity),   # 0 ~ -∞
    'energy': -np.sum(torques ** 2) * 0.01,   # 작은 값
    'contact': foot_contact_score,            # 0 ~ 1
}

# Weighted sum
total = sum(weight * r for weight, r in zip(weights, rewards.values()))
```

### 추천 weights
```python
weights = {
    'forward': 2.0,        # 주 목표
    'stability': 0.5,      # 보조 목표
    'energy': 0.1,         # 제약 조건
    'contact': 0.3,        # 보조 목표
}
```

---

## 해결책 5: Curriculum-Based Reward

### 원리
학습 진행에 따라 reward 비중 조정

```python
class CurriculumReward:
    def __init__(self):
        self.phase = 0  # 0: stand, 1: balance, 2: walk
        self.timestep = 0

    def get_reward(self, obs, action, info):
        if self.phase == 0:  # Phase 0: 서있기만
            return info['healthy_reward']

        elif self.phase == 1:  # Phase 1: 균형 유지
            return (
                info['healthy_reward'] +
                0.1 * np.tanh(info['x_velocity'])
            )

        else:  # Phase 2: 빠르게 걷기
            return (
                0.2 * info['healthy_reward'] +
                2.0 * np.tanh(info['x_velocity']) +
                info['contact_consistency']
            )

    def update_phase(self, success_rate):
        # 성공률 80% 이상이면 다음 단계
        if success_rate > 0.8:
            self.phase = min(self.phase + 1, 2)
```

---

## 해결책 6: Intrinsic Motivation (RND, ICM)

### 원리
외부 reward 외에 **탐색 보너스** 추가

```python
# Random Network Distillation (RND)
# 새로운 상태를 방문하면 보상
intrinsic_reward = predictor_error(obs)

total_reward = extrinsic_reward + 0.1 * intrinsic_reward
```

### 장점
- 제자리 걸음 회피
- 새로운 동작 탐색 유도

---

## 실전 추천 조합

### 🎯 최종 추천 설정

```python
class ImprovedReward:
    def __init__(self):
        # Normalized weights (모두 비슷한 스케일)
        self.forward_weight = 2.0
        self.healthy_weight = 0.2
        self.action_rate_weight = 0.1
        self.energy_weight = 0.01
        self.contact_weight = 0.5

    def compute(self, x_velocity, is_healthy, action_diff,
                energy, foot_contacts):

        # 1. Forward (Tanh 정규화)
        forward = self.forward_weight * np.tanh(x_velocity)

        # 2. Healthy (작게)
        healthy = self.healthy_weight if is_healthy else -2.0

        # 3. Smoothness
        action_rate_cost = self.action_rate_weight * np.sum(action_diff ** 2)

        # 4. Energy efficiency
        energy_cost = self.energy_weight * energy

        # 5. Contact consistency (교대로 걷기)
        contact_bonus = self.contact_weight * (
            1.0 if np.sum(foot_contacts) == 1 else 0.0
        )

        total = (
            forward + healthy + contact_bonus
            - action_rate_cost - energy_cost
        )

        return total, {
            'forward': forward,
            'healthy': healthy,
            'contact': contact_bonus,
            'action_rate_cost': action_rate_cost,
            'energy_cost': energy_cost,
        }
```

---

## 비교 표

| 방법 | Reward 범위 | 학습 안정성 | 구현 난이도 | 추천도 |
|------|------------|-----------|-----------|--------|
| Linear (현재) | Unbounded | ❌ 낮음 | ⭐ 쉬움 | ❌ |
| **Tanh/Sigmoid** | **Bounded** | **✅ 높음** | **⭐⭐ 쉬움** | **⭐⭐⭐** |
| Sparse+Dense | Mixed | ⚠️ 중간 | ⭐⭐ 중간 | ⭐⭐ |
| Relative Progress | Adaptive | ✅ 높음 | ⭐⭐⭐ 어려움 | ⭐⭐ |
| Multi-Objective | Bounded | ✅ 높음 | ⭐⭐ 중간 | ⭐⭐⭐ |
| Curriculum | Dynamic | ✅ 높음 | ⭐⭐⭐⭐ 어려움 | ⭐⭐ |

---

## 다음 단계

1. **즉시 시도:** Tanh 정규화 (v2 환경)
2. **추가:** Foot contact reward
3. **최적화:** Multi-objective balancing
4. **고급:** Curriculum learning (optional)

**예상 효과:**
- Reward explosion 해결 ✅
- 학습 안정성 향상 ✅
- 실제 전진 유도 ✅
