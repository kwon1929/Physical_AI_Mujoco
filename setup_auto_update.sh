#!/bin/bash
# 자동 업데이트 설정 스크립트

PROJECT_DIR="/Users/kwonhyuk/physical_ai_mujoco"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

echo "🤖 Humanoid Swarm Intelligence - 자동 업데이트 설정"
echo "================================================"

# 1. Twitter API 라이브러리 설치
echo ""
echo "📦 필요한 패키지 설치 중..."
source "$PROJECT_DIR/venv/bin/activate"
pip install tweepy python-dotenv

# 2. .env 파일 체크
echo ""
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "✅ .env 파일이 이미 존재합니다."
else
    echo "⚠️  .env 파일을 생성하세요:"
    echo "   cp .env.example .env"
    echo "   그 다음 Twitter API 키를 입력하세요."
fi

# 3. 수동 테스트
echo ""
echo "📝 수동 테스트:"
echo "   python daily_update.py --skip-twitter  # X 포스팅 없이 테스트"
echo "   python daily_update.py                 # 전체 실행"

# 4. 자동화 옵션 제공
echo ""
echo "⏰ 자동화 옵션:"
echo ""
echo "옵션 1: Cron (macOS/Linux)"
echo "   매일 오후 6시에 자동 업데이트:"
echo "   crontab -e"
echo "   그 다음 아래 줄 추가:"
echo "   0 18 * * * cd $PROJECT_DIR && $VENV_PYTHON daily_update.py"
echo ""
echo "옵션 2: 수동 실행"
echo "   원하는 시간에 직접:"
echo "   python daily_update.py"
echo ""
echo "✅ 설정 완료!"
