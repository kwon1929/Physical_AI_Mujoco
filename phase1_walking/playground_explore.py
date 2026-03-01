"""Week 4: Explore MuJoCo Menagerie humanoid models.

Run: python -m phase1_walking.playground_explore
     python -m phase1_walking.playground_explore --model g1
     python -m phase1_walking.playground_explore --model h1
     python -m phase1_walking.playground_explore --model berkeley
     python -m phase1_walking.playground_explore --model all
"""
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from phase1_walking.config import MENAGERIE_DIR

# Menagerie 휴머노이드 모델 경로
MODELS = {
    "g1": MENAGERIE_DIR / "unitree_g1" / "scene.xml",
    "h1": MENAGERIE_DIR / "unitree_h1" / "scene.xml",
    "berkeley": MENAGERIE_DIR / "berkeley_humanoid" / "scene.xml",
    "apollo": MENAGERIE_DIR / "apptronik_apollo" / "scene.xml",
    "talos": MENAGERIE_DIR / "pal_talos" / "scene.xml",
    "op3": MENAGERIE_DIR / "robotis_op3" / "scene.xml",
}


def get_model_info(xml_path: str) -> dict:
    """MuJoCo 모델의 기본 정보 추출."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 전체 질량 계산
    total_mass = sum(model.body_mass)

    return {
        "nq": model.nq,          # generalized positions
        "nv": model.nv,          # generalized velocities (DoF)
        "nu": model.nu,          # actuators
        "nbody": model.nbody,   # bodies
        "njnt": model.njnt,     # joints
        "total_mass": total_mass,
        "timestep": model.opt.timestep,
    }


def compare_models() -> None:
    """모든 휴머노이드 모델의 비교 테이블 출력."""
    print("\n--- Humanoid Model Comparison ---")
    print(f"  {'Model':<12} {'DoF':>5} {'Actuators':>10} {'Bodies':>8} {'Joints':>8} {'Mass (kg)':>10} {'dt':>8}")
    print(f"  {'─' * 65}")

    for name, xml_path in MODELS.items():
        if not xml_path.exists():
            print(f"  {name:<12} (not found: {xml_path})")
            continue

        info = get_model_info(str(xml_path))
        print(
            f"  {name:<12} {info['nv']:>5} {info['nu']:>10} "
            f"{info['nbody']:>8} {info['njnt']:>8} "
            f"{info['total_mass']:>10.2f} {info['timestep']:>8.4f}"
        )


def view_model(model_name: str, duration: float = 10.0) -> None:
    """MuJoCo 뷰어로 모델을 인터랙티브하게 시각화.

    Args:
        model_name: MODELS dict의 키
        duration: 시뮬레이션 시간 (초)
    """
    xml_path = MODELS[model_name]
    if not xml_path.exists():
        print(f"모델 파일을 찾을 수 없습니다: {xml_path}")
        return

    print(f"\n--- Viewing: {model_name} ---")
    info = get_model_info(str(xml_path))
    print(f"  DoF: {info['nv']}, Actuators: {info['nu']}, Mass: {info['total_mass']:.2f} kg")
    print(f"  MuJoCo 뷰어를 엽니다... (닫으려면 창을 닫거나 Ctrl+C)")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            # 작은 랜덤 토크를 적용하여 자연스러운 동작 확인
            data.ctrl = np.random.uniform(-0.1, 0.1, size=model.nu)
            mujoco.mj_step(model, data)
            viewer.sync()

            # 실시간 속도로 시뮬레이션
            elapsed = time.time() - start
            sim_time = data.time
            if sim_time > elapsed:
                time.sleep(sim_time - elapsed)

    print("  뷰어 종료.")


def try_playground() -> None:
    """MuJoCo Playground 사전학습 정책 시도 (optional)."""
    print("\n--- MuJoCo Playground ---")
    try:
        import playground
        print(f"  Playground version: {playground.__version__}")
        print("  사전학습된 locomotion 정책을 로드합니다...")
        # Playground API는 버전에 따라 다를 수 있음
        print("  (Playground 탐색은 별도 스크립트로 진행하세요)")
    except ImportError:
        print("  MuJoCo Playground가 설치되지 않았습니다.")
        print("  설치: pip install playground")
        print("  참고: JAX가 필요합니다. playground.mujoco.org 참조")


def main():
    parser = argparse.ArgumentParser(description="Explore humanoid models from MuJoCo Menagerie")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=list(MODELS.keys()) + ["all", "playground"],
        help="Which model to explore (default: all for comparison table)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(" Phase 1, Week 4: MuJoCo Menagerie Exploration")
    print("=" * 60)

    if args.model == "all":
        compare_models()
        print("\n  특정 모델을 시각화하려면:")
        print("  python -m phase1_walking.playground_explore --model g1")
    elif args.model == "playground":
        try_playground()
    else:
        view_model(args.model)


if __name__ == "__main__":
    main()
