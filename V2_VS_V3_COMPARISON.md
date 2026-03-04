# G1 Walking: V2 vs V3 비교

## V2 실패 원인 분석

**증상:**
- Episode length: 39 steps (매우 짧음)
- Forward distance: 0.12m (거의 움직이지 않음)
- Total reward: -26.5 (음수!)
- Final pelvis z: 0.28m (healthy_z_range 0.3m 밑으로 떨어져 종료)

**근본 원인:**
1. **과도한 페널티**: Action rate (0.1) + Energy (0.001) + Ctrl (0.01) 합이 forward reward보다 큼
2. **Tanh 정규화 문제**: Forward reward = 2.0 × tanh(v) → 최대 2.0으로 제한
3. **즉각적인 붕괴**: 로봇이 40 step 만에 넘어짐

---

## V3 "Lean Walking" 개선사항

### 1. 🎯 Reward Function 단순화

**V2 (복잡함):**
```python
forward_reward = 2.0 * np.tanh(x_velocity)      # 최대 2.0
healthy_reward = 0.2
action_rate_cost = 0.1 * Σ(action_diff²)        # 페널티
energy_cost = 0.001 * Σ|τ·ω|                    # 페널티
ctrl_cost = 0.01 * Σ(action²)                   # 페널티
contact_stability = 0.5 (조건부)
contact_consistency = 0.3 (조건부)

총 reward = forward + healthy + contacts - penalties
```

**문제점:**
- Forward reward가 너무 작음 (최대 2.0)
- 페널티 합이 forward보다 클 수 있음
- → 총 reward 음수!

---

**V3 (단순함):**
```python
forward_reward = 10.0 * x_velocity              # 선형, 제한 없음
healthy_reward = 5.0                            # 높음
ctrl_cost = 0.001 * Σ(action²)                  # 최소화

총 reward = forward + healthy - ctrl_cost
```

**개선점:**
- Forward reward가 선형이라 속도에 비례해 증가
- Healthy reward가 높아서 서있기만 해도 큰 보상
- 페널티는 거의 없음 (0.001)
- → 총 reward 양수 보장!

---

### 2. ⚡ Action Scale 증가

| 버전 | Action Scale | 설명 |
|------|-------------|------|
| V2 | 0.5 | 보통 |
| **V3** | **0.7** | **로봇이 더 큰 힘으로 땅을 밀 수 있음** |

**효과:**
- 더 강한 움직임 가능
- 걷기 위한 충분한 토크 제공

---

### 3. 🛡️ Termination Condition 완화

| 버전 | Healthy Z Range | 설명 |
|------|----------------|------|
| V2 | (0.3, 1.2) | V2 결과: 0.28m에서 종료 (경계 근처) |
| **V3** | **(0.25, 1.5)** | **더 넓은 범위, 초반 붕괴 방지** |

**효과:**
- 로봇이 약간 낮게 앉아도 종료 안됨
- 학습 초기에 더 많은 탐색 가능

---

### 4. 🧠 Observation 강화

**V2:**
```python
obs = [qpos[2:], qvel, prev_action]  # 98 dims
```

**V3:**
```python
obs = [qpos[2:], qvel, foot_contacts]  # 71 dims
```

**차이점:**
- V3에서 `prev_action` 제거 (단순화)
- `foot_contacts` 명시적 포함 (발 접촉 정보)
- qvel (속도) 전체 포함 → 로봇이 현재 속도를 정확히 인지

**효과:**
- 속도 정보로 전진 방향 파악
- 발 접촉으로 균형 유지 가능

---

## Reward 범위 비교

### V2 Reward 계산 예시:
```
가정: x_velocity = 0.5 m/s, healthy, action²=0.5, action_diff²=0.3, energy=1.0

forward_reward = 2.0 × tanh(0.5) = 2.0 × 0.46 = 0.92
healthy_reward = 0.2
ctrl_cost = 0.01 × 0.5 = 0.005
action_rate_cost = 0.1 × 0.3 = 0.03
energy_cost = 0.001 × 1.0 = 0.001

총 reward = 0.92 + 0.2 - 0.005 - 0.03 - 0.001 = 1.08
```
→ 양수이지만 **매우 작음**. 만약 x_velocity가 작으면 음수 가능.

---

### V3 Reward 계산 예시:
```
가정: x_velocity = 0.5 m/s, healthy, action²=0.5

forward_reward = 10.0 × 0.5 = 5.0
healthy_reward = 5.0
ctrl_cost = 0.001 × 0.5 = 0.0005

총 reward = 5.0 + 5.0 - 0.0005 ≈ 10.0
```
→ **항상 양수이고 큼!** 서있기만 해도 5.0, 걸으면 10.0+

---

## 예상 결과 비교

| 지표 | V2 (실패) | V3 (예상) |
|------|----------|----------|
| Episode Length | 39 steps | >= 200 steps |
| Forward Distance | 0.12m | >= 3.0m |
| Total Reward | -26.5 | +1000+ |
| 최종 Pelvis Z | 0.28m | ~0.8m |
| 행동 | 즉시 넘어짐 | 안정적으로 걷기 |

---

## 학습 전략

### V2 문제:
- 페널티가 너무 커서 로봇이 "아무것도 하지 않기" 학습
- Forward reward가 작아서 걷기 동기 부족
- 빨리 넘어지면 학습 데이터 부족

### V3 접근:
1. **단계 1: 서있기 학습** (Healthy reward 5.0)
   - 로봇이 균형 유지하는 법 학습
2. **단계 2: 전진 시도** (Forward reward 10.0)
   - 서있기보다 걷기가 더 높은 보상
3. **단계 3: 걷기 최적화**
   - 속도를 높여서 더 많은 forward reward

---

## 사용 방법

### V3 학습 시작:
```bash
# 학습
./venv/bin/python -m phase1_walking.train_v3

# TensorBoard
tensorboard --logdir logs/ppo_g1_v3

# 평가
./venv/bin/python -m phase1_walking.evaluate_v3 --record
open videos/
```

### 성공 기준:
- [ ] Episode length >= 200 steps
- [ ] Forward distance >= 3.0m
- [ ] Total reward >= 1000
- [ ] 시각적으로 걷는 모습 확인

---

## 요약

| 항목 | V2 | V3 |
|------|----|----|
| **철학** | 복잡한 보상, 여러 제약 | 단순한 보상, 걷기 집중 |
| **Forward Reward** | Tanh (최대 2.0) | 선형 (10.0 × v) |
| **Healthy Reward** | 0.2 (작음) | 5.0 (큼) |
| **페널티** | 많음 (action rate, energy 등) | 거의 없음 (ctrl만) |
| **Action Scale** | 0.5 | 0.7 |
| **Termination** | (0.3, 1.2) | (0.25, 1.5) |
| **결과** | ❌ 39 steps, 0.12m | ⏳ 테스트 중 |

---

**🎉 V3로 드디어 G1이 걷기를 배울 것입니다!**
