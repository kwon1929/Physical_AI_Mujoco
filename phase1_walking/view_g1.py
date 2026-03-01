"""Unitree G1 시각화: 키프레임으로 서있는 모습 확인.

Run: python -m phase1_walking.view_g1
     python -m phase1_walking.view_g1 --mode stand    (서있기)
     python -m phase1_walking.view_g1 --mode free      (자유 낙하)
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from phase1_walking.config import MENAGERIE_DIR

G1_SCENE = str(MENAGERIE_DIR / "unitree_g1" / "scene.xml")


def view_standing(duration: float = 30.0) -> None:
    """G1이 'stand' 키프레임으로 서있는 모습을 표시.

    position control로 관절을 고정하기 때문에 안정적으로 서있음.
    """
    model = mujoco.MjModel.from_xml_path(G1_SCENE)
    data = mujoco.MjData(model)

    # "stand" 키프레임 적용
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # 모델 정보 출력
    print(f"\n--- Unitree G1 Model Info ---")
    print(f"  Joints (DoF):  {model.nv}")
    print(f"  Actuators:     {model.nu}")
    print(f"  Bodies:        {model.nbody}")
    print(f"  Total mass:    {sum(model.body_mass):.2f} kg")
    print(f"  Pelvis height: {data.qpos[2]:.3f} m")
    print(f"\n  Actuator names:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"    [{i:2d}] {name}")

    print(f"\n  MuJoCo 뷰어를 엽니다. 마우스로 시점 회전 가능.")
    print(f"  창을 닫으면 종료됩니다.\n")

    mujoco.viewer.launch(model, data)


def view_freefall(duration: float = 30.0) -> None:
    """G1을 제어 없이 놓아두면 어떻게 되는지 확인.

    ctrl을 0으로 설정하면 position control 목표가 0이 되어
    관절이 풀려서 넘어짐. (학습 전 기본 상태)
    """
    model = mujoco.MjModel.from_xml_path(G1_SCENE)
    data = mujoco.MjData(model)

    # stand 키프레임으로 초기 자세 설정
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # ctrl을 0으로 → 관절이 풀림
    data.ctrl[:] = 0

    print(f"\n  ctrl=0으로 설정 (제어 없음). G1이 넘어지는 모습을 확인하세요.")
    print(f"  이것이 RL로 학습해야 하는 이유입니다!\n")

    mujoco.viewer.launch(model, data)


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 시각화")
    parser.add_argument(
        "--mode", type=str, default="stand",
        choices=["stand", "free"],
        help="stand: 키프레임으로 서있기, free: 제어없이 자유낙하",
    )
    parser.add_argument("--duration", type=float, default=30.0, help="시뮬레이션 시간 (초)")
    args = parser.parse_args()

    print("=" * 50)
    print(" Unitree G1 Visualization")
    print("=" * 50)

    if args.mode == "stand":
        view_standing(args.duration)
    else:
        view_freefall(args.duration)


if __name__ == "__main__":
    main()
