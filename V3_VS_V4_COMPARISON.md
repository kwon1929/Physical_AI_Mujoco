## G1 Walking: V3 vs V4 비교 (MiniCheetah 기법 적용)

## V4 핵심 개선사항 (MiniCheetah 영감)

### 1. 🎯 지수 기반 Forward Reward

**V3 (선형):**
```python
forward_reward = 10.0 * x_velocity  # 속도가 높을수록 무한정 증가
```

**문제점:**
- 너무 빠르게 달리려다 넘어질 수 있음
- 최적 속도 개념 없음

---

**V4 (지수 기반):**
```python
target_velocity = 1.0  # m/s
velocity_error = (x_velocity - target_velocity) ** 2
forward_reward = 10.0 * exp(-velocity_error)
```

**장점:**
- 목표 속도 1.0 m/s에서 최대 보상 (10.0)
- 너무 빠르거나 느리면 페널티
- 안정적인 걷기 유도

**Reward 그래프:**
```
Reward
  10 |     ***
     |   **   **
   5 |  *       *
     | *         *
   0 |*___________*___ Velocity
     0   0.5  1.0  1.5  2.0

최적점: 1.0 m/s에서 reward = 10.0
```

---

### 2. ⚖️ 자세 안정성 보상 (NEW!)

**V3:**
- 자세 안정성 보상 없음
- Healthy reward만 있음 (높이만 체크)

**V4:**
```python
# Roll/Pitch 각도 최소화
roll_penalty = roll²
pitch_penalty = pitch²
stability_reward = 5.0 * (exp(-roll²) + exp(-pitch²)) / 2
```

**효과:**
- Roll/Pitch 각도가 0에 가까울수록 높은 보상
- 똑바로 서있기 유도
- 넘어짐 방지

---

### 3. 🛡️ Roll/Pitch Termination (NEW!)

**V3:**
```python
is_healthy = (0.25 < pelvis_z < 1.5)  # 높이만 체크
```

**문제점:**
- 높이는 괜찮지만 옆으로 넘어져도 감지 못함

---

**V4:**
```python
is_healthy = (
    (0.25 < pelvis_z < 1.5)  # 높이
    AND abs(roll) < 0.8      # Roll < 45°
    AND abs(pitch) < 0.8     # Pitch < 45°
)
```

**효과:**
- 넘어짐을 명확히 감지
- 자세 안정성 강제

---

## Reward Function 비교

### V3 Reward:
```python
reward = (
    10.0 * x_velocity      # 선형 (0~10+)
    + 5.0                  # 생존 (5.0)
    - 0.001 * ctrl_cost    # 토크 (~0.0)
)
# 예상 범위: 5~15+
```

### V4 Reward:
```python
reward = (
    10.0 * exp(-(v - 1.0)²)  # 지수 (0~10)
    + 5.0 * (exp(-roll²) + exp(-pitch²))/2  # 자세 (0~5)
    + 3.0                    # 생존 (3.0)
    - 0.01 * ctrl_cost       # 토크 (~0.0)
)
# 예상 범위: 3~18
```

---

## 예상 성능 비교

| 지표 | V3 (4M) | V4 (4M 예상) | 개선 |
|------|---------|-------------|------|
| Episode Length | 127 steps | **150+ steps** | +18% |
| Forward Distance | 0.94m | **1.5m+** | +60% |
| Total Reward | +1558 | **+2000+** | +28% |
| 속도 제어 | ❌ 무한정 가속 | ✅ 최적 속도 유지 | ⭐ |
| 자세 안정성 | ⚠️ 암시적 | ✅ 명시적 보상 | ⭐ |
| Termination | Z만 | Z + Roll + Pitch | ⭐ |

---

## Observation Space

| V3 | V4 | 차이 |
|----|----|----|
| 71 dims | 71 dims | 동일 |
| qpos + qvel + contacts | qpos + qvel + contacts | 동일 |

*V4는 reward function만 변경, observation은 동일*

---

## 학습 전략 차이

### V3 학습:
1. 서있기 학습 (Healthy reward 5.0)
2. 빠르게 전진 시도 (Linear reward)
3. → 너무 빠르게 달리다 넘어질 가능성

### V4 학습:
1. 서있기 학습 (Healthy reward 3.0)
2. **자세 안정성 학습** (Stability reward 5.0) ⭐
3. **목표 속도 달성** (Target 1.0 m/s) ⭐
4. → 안정적이고 제어된 걷기

---

## 실제 Reward 계산 예시

### V3:
```
가정: x_velocity = 0.5 m/s, healthy, ctrl=0.5

forward = 10.0 × 0.5 = 5.0
healthy = 5.0
ctrl_cost = 0.001 × 0.5 = 0.0005

총 reward = 5.0 + 5.0 - 0.0005 ≈ 10.0
```

### V4:
```
가정: x_velocity = 0.5 m/s, roll=0.1, pitch=0.1, ctrl=0.5

forward = 10.0 × exp(-(0.5-1.0)²) = 10.0 × exp(-0.25) = 7.8
stability = 5.0 × (exp(-0.01) + exp(-0.01))/2 = 4.95
healthy = 3.0
ctrl_cost = 0.01 × 0.5 = 0.005

총 reward = 7.8 + 4.95 + 3.0 - 0.005 ≈ 15.7
```

**V4가 더 높은 reward!** (자세 안정성 보상 때문)

---

### 목표 속도 달성 시 (1.0 m/s):
```
forward = 10.0 × exp(0) = 10.0  (최대!)
stability = 4.95
healthy = 3.0

총 reward ≈ 18.0  (V3의 15.0보다 높음)
```

---

## 사용 방법

### V3 평가 (참고용):
```bash
./venv/bin/python -m phase1_walking.evaluate_v3 --record
```

### V4 학습 (V3 4M 완료 후):
```bash
# 학습
./venv/bin/python -m phase1_walking.train_v4

# 평가
./venv/bin/python -m phase1_walking.evaluate_v4 --record

# TensorBoard
tensorboard --logdir logs/ppo_g1_v4
```

---

## 핵심 차이 요약

| 항목 | V3 | V4 |
|------|----|----|
| **철학** | 빠르게 걷기 | 안정적으로 걷기 |
| **Forward Reward** | 선형 (무한) | 지수 (목표 속도) |
| **자세 보상** | ❌ 없음 | ✅ Roll/Pitch 명시적 |
| **Termination** | Z만 | Z + Roll + Pitch |
| **Healthy Reward** | 5.0 | 3.0 (자세 보상에 집중) |
| **제어** | 속도 제어 없음 | 속도 제어 있음 |
| **예상 결과** | ~127 steps, 0.94m | 150+ steps, 1.5m+ |

---

**🎉 V4는 MiniCheetah의 검증된 기법을 적용한 안정적인 보행 환경입니다!**
