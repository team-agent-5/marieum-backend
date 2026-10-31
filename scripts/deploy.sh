#!/usr/bin/env bash
# Marieum Backend — NCP 서버 배포 스크립트
#
# 사용법:
#   bash scripts/deploy.sh
#
# 전제:
#   - NCP VPN 접속 상태
#   - SSH 비밀번호 입력 2회 요구됨 (rsync + ssh)
#     → SSH 키 등록하면 이후부터 비번 없이 가능:
#       ssh-copy-id -p 22 ncloud@10.1.1.11
#
# 서버 측 전제:
#   - python3 (3.11+ 권장) 설치
#   - 아웃바운드 pip 가능 (사내망 pypi 미러 or 외부)
#   - NCP ACG(방화벽)에서 8000/tcp inbound 허용 — 콘솔에서 사용자가 설정

set -euo pipefail

HOST="${HOST:-10.1.1.11}"
SSH_USER="${SSH_USER:-ncloud}"
PORT="${PORT:-22}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/marieum_ncp}"
REMOTE_DIR="${REMOTE_DIR:-/home1/${SSH_USER}/marieum-backend}"
APP_PORT="${APP_PORT:-8000}"

SSH_OPTS=(-i "${SSH_KEY}" -p "${PORT}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "────────────────────────────────────────────────────"
echo "📤 Marieum Backend Deploy"
echo "   Local : ${HERE}"
echo "   Remote: ${SSH_USER}@${HOST}:${REMOTE_DIR}"
echo "   Port  : ${APP_PORT}"
echo "────────────────────────────────────────────────────"
echo ""

# 1. 코드 전송
echo "[1/3] 코드 전송 (rsync)..."
rsync -avz --delete \
  --exclude '.venv*' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.env.local' \
  --exclude 'logs' \
  --exclude 'tmp' \
  --exclude '.DS_Store' \
  --exclude '*.wav' --exclude '*.mp3' --exclude '*.m4a' --exclude '*.webm' \
  -e "ssh ${SSH_OPTS[*]}" \
  "${HERE}/" \
  "${SSH_USER}@${HOST}:${REMOTE_DIR}/"

echo ""
echo "[2/3] 원격 설정 및 서버 재시작..."

ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" bash -s << ENDSSH
set -euo pipefail
cd "${REMOTE_DIR}"

# 가상환경
if [ ! -d .venv ]; then
  echo "  → 가상환경 생성"
  python3 -m venv .venv
fi
source .venv/bin/activate

# 의존성 (API 스웨거 + DB + 마이그레이션 + seed)
echo "  → 의존성 설치"
pip install -q --upgrade pip
pip install -q \\
  fastapi \\
  'uvicorn[standard]' \\
  'sqlalchemy[asyncio]>=2.0' \\
  'asyncpg>=0.29' \\
  'alembic>=1.13' \\
  'pgvector>=0.3' \\
  'psycopg2-binary' \\
  python-multipart \\
  pydantic \\
  'pydantic-settings>=2.5' \\
  pyjwt \\
  python-dotenv

# DB 마이그레이션 적용
echo "  → alembic upgrade head"
PYTHONPATH=src ./.venv/bin/alembic upgrade head || {
  echo "  ⚠ alembic 실패. .env의 DATABASE_URL과 docker DB 상태 확인 필요"
}

# 기존 프로세스 종료
if pgrep -f 'uvicorn app.main' > /dev/null; then
  echo "  → 기존 프로세스 종료"
  pkill -f 'uvicorn app.main' || true
  sleep 1
fi

# 로그 디렉터리
mkdir -p logs

# 백그라운드 실행 (nohup).
# cwd 는 프로젝트 루트 유지 — pydantic-settings 가 .env 를 찾아야 하므로.
# TZ=Asia/Seoul — 로그/에러 메시지 KST 표시 (저장은 timestamptz UTC 유지).
TZ=Asia/Seoul nohup ./.venv/bin/uvicorn app.main:app \\
  --host 0.0.0.0 \\
  --port ${APP_PORT} \\
  --app-dir src \\
  --log-level info \\
  > logs/uvicorn.log 2>&1 &

sleep 2

# 상태 확인
if pgrep -f 'uvicorn app.main' > /dev/null; then
  echo "  ✓ 서버 시작됨 (PID: \$(pgrep -f 'uvicorn app.main'))"
else
  echo "  ✗ 서버 시작 실패. 로그 확인:"
  tail -20 logs/uvicorn.log
  exit 1
fi

# 헬스체크
if command -v curl >/dev/null 2>&1; then
  echo "  → /healthz 확인"
  sleep 1
  curl -sS "http://localhost:${APP_PORT}/healthz" || echo "  (헬스체크 응답 없음, 로그 확인)"
  echo ""
fi
ENDSSH

echo ""
echo "[3/3] 배포 완료!"
echo ""
echo "────────────────────────────────────────────────────"
echo "📡 내부망 접속 (NCP VPN 접속 상태)"
echo "   Swagger  : http://${HOST}:${APP_PORT}/docs"
echo "   OpenAPI  : http://${HOST}:${APP_PORT}/openapi.json"
echo "   Health   : http://${HOST}:${APP_PORT}/healthz"
echo ""

# Cloudflare Tunnel 상태 확인 (서버 ~/.marieum-tunnel/url.txt)
TUNNEL_URL=$(ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" \
  'cat $HOME/.marieum-tunnel/url.txt 2>/dev/null' || true)
if [ -n "${TUNNEL_URL}" ]; then
  echo "🌐 외부 공개 URL (Cloudflare Tunnel, VPN 불필요)"
  echo "   Swagger  : ${TUNNEL_URL}/docs"
  echo "   OpenAPI  : ${TUNNEL_URL}/openapi.json"
  echo "   Health   : ${TUNNEL_URL}/healthz"
else
  echo "🌐 외부 공개 (Cloudflare Tunnel)"
  echo "   현재 꺼져 있음. 띄우려면:  make tunnel-up"
fi
echo ""
echo "📜 로그   :  make tunnel-logs  /  ssh … tail -f ${REMOTE_DIR}/logs/uvicorn.log"
echo "🛑 종료   :  make tunnel-down  /  ssh … 'pkill -f \"uvicorn app.main\"'"
echo "────────────────────────────────────────────────────"
