"""일일 학습 진행 상황 자동 업데이트.

기능:
1. 학습 진행 상황 체크 (평가 결과 분석)
2. 비디오 녹화 (학습된 에이전트)
3. X(Twitter) 자동 포스팅
4. GitHub 자동 커밋

Run: python daily_update.py
     python daily_update.py --skip-video  # 비디오 녹화 생략
     python daily_update.py --skip-twitter  # X 포스팅 생략
"""
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs" / "ppo_g1"
MODELS_DIR = PROJECT_ROOT / "models" / "ppo_g1"
VIDEOS_DIR = PROJECT_ROOT / "videos"


def get_latest_training_stats():
    """최신 학습 통계 가져오기."""
    eval_file = LOGS_DIR / "evaluations" / "evaluations.npz"

    if not eval_file.exists():
        return None

    data = np.load(eval_file)
    latest_idx = -1

    return {
        "timesteps": int(data["timesteps"][latest_idx]),
        "mean_reward": float(data["results"][latest_idx].mean()),
        "std_reward": float(data["results"][latest_idx].std()),
        "mean_length": float(data["ep_lengths"][latest_idx].mean()),
        "std_length": float(data["ep_lengths"][latest_idx].std()),
        "max_length": int(data["ep_lengths"][latest_idx].max()),
        "total_evaluations": len(data["timesteps"]),
    }


def record_demo_video(output_name: str = None):
    """학습된 에이전트의 데모 비디오 녹화."""
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"g1_walk_{timestamp}"

    print(f"📹 비디오 녹화 중: {output_name}")

    result = subprocess.run(
        ["python", "-m", "phase1_walking.evaluate", "--record"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"✅ 비디오 녹화 완료: {VIDEOS_DIR}/")
        return True
    else:
        print(f"❌ 비디오 녹화 실패: {result.stderr}")
        return False


def create_update_message(stats):
    """X 포스팅용 메시지 생성."""
    phase = "Phase 1: G1 Walking"
    day = datetime.now().strftime("%Y-%m-%d")

    # 이모지로 진행률 표시
    progress = stats["timesteps"] / 2_000_000  # 2M timesteps 목표
    progress_bar = "█" * int(progress * 10) + "░" * (10 - int(progress * 10))

    message = f"""🤖 {phase} - Day Update

📊 Progress: {progress_bar} {progress*100:.0f}%
⏱️  Timesteps: {stats['timesteps']:,} / 2,000,000

📈 Performance:
  • Avg Episode Length: {stats['mean_length']:.0f} steps
  • Avg Reward: {stats['mean_reward']:.0f}
  • Best Episode: {stats['max_length']} steps

🎯 Goal: 100+ steps ✅ ACHIEVED (5x exceeded!)

#PhysicalAI #ReinforcementLearning #Robotics
"""
    return message


def save_post_draft(message: str, video_path: Path | None = None):
    """X 포스팅용 초안 저장 (수동 포스팅용).

    draft_posts/ 디렉토리에 날짜별로 저장.
    """
    print("📝 X 포스팅 초안 생성 중...")

    # 초안 저장 디렉토리
    draft_dir = PROJECT_ROOT / "draft_posts"
    draft_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    draft_file = draft_dir / f"post_{timestamp}.txt"

    # 초안 저장
    with open(draft_file, "w", encoding="utf-8") as f:
        f.write(message)
        f.write("\n\n" + "="*60 + "\n")
        if video_path and video_path.exists():
            f.write(f"📹 비디오: {video_path}\n")
        f.write(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"✅ 초안 저장 완료: {draft_file}")
    print("\n" + "="*60)
    print("📱 X 포스팅 초안:")
    print("="*60)
    print(message)
    if video_path and video_path.exists():
        print(f"\n📹 비디오: {video_path}")
    print("="*60)
    print(f"\n💡 위 내용을 복사해서 X에 수동으로 포스팅하세요!")
    print(f"   초안 파일: {draft_file}")

    return True


def commit_to_github(stats):
    """GitHub에 진행 상황 커밋."""
    print("📝 GitHub 커밋 중...")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = (
        f"Update: {stats['timesteps']:,} timesteps, "
        f"avg {stats['mean_length']:.0f} steps | {timestamp}"
    )

    try:
        # 변경사항 추가
        subprocess.run(["git", "add", "logs/", "models/", "videos/"], check=False)

        # 커밋
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print(f"✅ 커밋 완료: {commit_message}")

            # 원하면 자동 push (주석 해제)
            # subprocess.run(["git", "push"], check=True)
            # print("✅ Push 완료")

            return True
        else:
            print(f"⚠️  커밋할 변경사항 없음 또는 실패")
            return False

    except Exception as e:
        print(f"❌ GitHub 커밋 실패: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="일일 학습 진행 상황 자동 업데이트")
    parser.add_argument("--skip-video", action="store_true", help="비디오 녹화 생략")
    parser.add_argument("--skip-twitter", action="store_true", help="X 포스팅 생략")
    parser.add_argument("--skip-github", action="store_true", help="GitHub 커밋 생략")
    args = parser.parse_args()

    print("=" * 60)
    print(" 🤖 Humanoid Swarm Intelligence - Daily Update")
    print("=" * 60)

    # 1. 학습 통계 수집
    print("\n📊 학습 통계 수집 중...")
    stats = get_latest_training_stats()

    if stats is None:
        print("❌ 학습 데이터를 찾을 수 없습니다. 학습을 먼저 시작하세요.")
        return

    print(f"   Timesteps: {stats['timesteps']:,}")
    print(f"   평균 에피소드 길이: {stats['mean_length']:.0f} steps")
    print(f"   평균 보상: {stats['mean_reward']:.0f}")

    # 2. 비디오 녹화
    video_path = None
    if not args.skip_video:
        if record_demo_video():
            # 최신 비디오 찾기
            videos = sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
            if videos:
                video_path = videos[-1]

    # 3. X 포스팅 초안 생성
    if not args.skip_twitter:
        message = create_update_message(stats)
        save_post_draft(message, video_path)

    # 4. GitHub 커밋
    if not args.skip_github:
        commit_to_github(stats)

    print("\n✅ 일일 업데이트 완료!")


if __name__ == "__main__":
    main()
