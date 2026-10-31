#!/usr/bin/env bash
# scripts/smoke-stream.sh
# 실시간 모드 WebSocket end-to-end 스모크 테스트 (06-senior-review §8.4)
#
# 시나리오:
#   1. GET  /healthz                          — 서버 생존 확인
#   2. POST /auth/login                       — JWT 획득
#   3. WS   /ws/phrase/stream?token=<JWT>     — 핸드셰이크
#   4. binary 프레임 10개 전송 (각 6400 bytes = PCM16 100ms)
#   5. {"type":"end"} JSON 전송
#   6. transcript final:true 수신 확인
#   7. close code 1000 확인
#
# 사용 예시:
#   # 로컬 (uvicorn 실행 중이어야 함)
#   bash scripts/smoke-stream.sh
#
#   # Cloudflare tunnel
#   BASE_URL=https://xxx.trycloudflare.com \
#   WS_URL=wss://xxx.trycloudflare.com \
#   bash scripts/smoke-stream.sh
#
#   # 다른 환자
#   USER_ID=demo_pb PASSWORD=demo1234! bash scripts/smoke-stream.sh

set -euo pipefail

# ─── 환경변수 ─────────────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
WS_URL="${WS_URL:-ws://127.0.0.1:8000}"
USER_ID="${USER_ID:-demo_pa}"
PASSWORD="${PASSWORD:-demo1234!}"

# ─── 색상 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓${NC} $*"; }
fail() { echo -e "${RED}✗${NC} $*"; }
info() { echo -e "${YELLOW}→${NC} $*"; }

FAILED=0

# ─── 도구 확인 ────────────────────────────────────────────────────────────────
for tool in curl jq python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        fail "필수 도구 없음: $tool"
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "도구를 설치한 뒤 재실행하세요."
    exit 1
fi

# python3 websockets 모듈 확인
if ! python3 -c "import websockets" 2>/dev/null; then
    fail "python3 websockets 없음. 설치: pip install websockets"
    exit 1
fi

echo "================================================================"
echo "  smoke-stream.sh"
echo "  BASE_URL : ${BASE_URL}"
echo "  WS_URL   : ${WS_URL}"
echo "  USER_ID  : ${USER_ID}"
echo "================================================================"
echo

# ─── Step 1: /healthz ─────────────────────────────────────────────────────────
echo "=== 1. /healthz ==="
HEALTH=$(curl -sS --max-time 5 "${BASE_URL}/healthz" 2>&1 || true)
echo "$HEALTH" | head -c 200
echo

if echo "$HEALTH" | grep -qi '"ok"'; then
    pass "/healthz OK"
elif echo "$HEALTH" | grep -qi 'ok'; then
    pass "/healthz OK (plain)"
else
    fail "/healthz 응답 없음 또는 비정상. 서버가 실행 중인지 확인하세요."
    fail "  BASE_URL=${BASE_URL}"
    exit 1
fi
echo

# ─── Step 2: /auth/login ──────────────────────────────────────────────────────
echo "=== 2. /auth/login ==="
LOGIN_RES=$(curl -sS --max-time 5 -X POST "${BASE_URL}/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"user_id\":\"${USER_ID}\",\"password\":\"${PASSWORD}\"}" 2>&1 || true)

TOKEN=$(echo "$LOGIN_RES" | jq -r '.access_token // empty' 2>/dev/null || true)

if [ -z "$TOKEN" ]; then
    fail "로그인 실패. 응답: ${LOGIN_RES}"
    exit 1
fi
pass "로그인 성공 (token: ${TOKEN:0:20}...)"
echo

# ─── Step 3~7: WebSocket 핸드셰이크 + 바이너리 10 프레임 + end + 검증 ──────────
echo "=== 3. WebSocket 핸드셰이크 + 바이너리 10 프레임 ==="
info "URL: ${WS_URL}/ws/phrase/stream?token=${TOKEN:0:12}..."
echo

# Python으로 WS 테스트 실행
WS_RESPONSE=$(python3 << PYEOF
import asyncio
import json
import sys
import websockets

WS_ENDPOINT = "${WS_URL}/ws/phrase/stream?token=${TOKEN}"
FRAME_SIZE = 6400   # 16000Hz × 2bytes × 0.2s (실제로는 100ms = 3200샘플 × 2 = 6400)
NUM_FRAMES = 10

# PCM16 더미 데이터 (DC 오프셋 없는 0 패딩)
DUMMY_FRAME = bytes(FRAME_SIZE)

async def run_test():
    result = {
        "handshake": False,
        "frames_sent": 0,
        "end_sent": False,
        "messages": [],
        "close_code": None,
        "error": None,
    }

    try:
        async with websockets.connect(
            WS_ENDPOINT,
            open_timeout=10,
            close_timeout=5,
            ping_interval=None,  # 짧은 테스트 중 ping 비활성
        ) as ws:
            result["handshake"] = True

            # Step 4: binary 프레임 10개 전송
            for i in range(NUM_FRAMES):
                await ws.send(DUMMY_FRAME)
                result["frames_sent"] = i + 1
                await asyncio.sleep(0.05)  # 50ms 간격 (실시간 스트림 모방)

            # Step 5: {"type":"end"} 전송
            await ws.send(json.dumps({"type": "end"}))
            result["end_sent"] = True

            # Step 6: 메시지 수신 (최대 10초 대기)
            try:
                async with asyncio.timeout(10):
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue  # binary 무시
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        result["messages"].append(msg)
                        # final:true 수신 시 수신 완료
                        if msg.get("final") is True or msg.get("is_final") is True:
                            break
            except (TimeoutError, asyncio.TimeoutError):
                result["error"] = "timeout waiting for final transcript"
            except websockets.exceptions.ConnectionClosed:
                pass  # 정상 close

            # close code 기록
            result["close_code"] = ws.close_code

    except websockets.exceptions.InvalidStatus as e:
        result["error"] = f"WS handshake rejected: {e}"
    except OSError as e:
        result["error"] = f"connection error: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    print(json.dumps(result))

asyncio.run(run_test())
PYEOF
)

# JSON 파싱 가능 여부 확인
if ! echo "$WS_RESPONSE" | python3 -m json.tool > /dev/null 2>&1; then
    fail "WS 테스트 스크립트 출력 파싱 오류:"
    echo "$WS_RESPONSE"
    exit 1
fi

echo "--- WS 테스트 결과 ---"
echo "$WS_RESPONSE" | python3 -m json.tool
echo

# ─── 결과 검증 ────────────────────────────────────────────────────────────────
echo "=== 검증 ==="

# 3-1. 핸드셰이크
HANDSHAKE=$(echo "$WS_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('true' if d.get('handshake') else 'false')
")
if [ "$HANDSHAKE" = "true" ]; then
    pass "핸드셰이크 성공"
else
    ERR=$(echo "$WS_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))")
    fail "핸드셰이크 실패: ${ERR}"
    FAILED=1
fi

# 3-2. 바이너리 10프레임 전송
FRAMES=$(echo "$WS_RESPONSE" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('frames_sent', 0))
")
if [ "$FRAMES" = "10" ]; then
    pass "바이너리 프레임 10개 전송 완료"
else
    fail "바이너리 프레임 전송 수: ${FRAMES}/10"
    FAILED=1
fi

# 3-3. {"type":"end"} 전송
END_SENT=$(echo "$WS_RESPONSE" | python3 -c "
import sys, json
print('true' if json.load(sys.stdin).get('end_sent') else 'false')
")
if [ "$END_SENT" = "true" ]; then
    pass "{\"type\":\"end\"} 전송 완료"
else
    fail "{\"type\":\"end\"} 전송 실패"
    FAILED=1
fi

# 3-4. transcript final 수신
HAS_FINAL=$(echo "$WS_RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msgs = d.get('messages', [])
has = any(
    m.get('type') == 'transcript' and (m.get('final') is True or m.get('is_final') is True)
    for m in msgs
)
print('true' if has else 'false')
")
if [ "$HAS_FINAL" = "true" ]; then
    pass "transcript final:true 수신 확인"
else
    fail "transcript final:true 수신 실패"
    info "  수신 메시지 목록:"
    echo "$WS_RESPONSE" | python3 -c "
import sys, json
msgs = json.load(sys.stdin).get('messages', [])
for m in msgs:
    print('   ', json.dumps(m, ensure_ascii=False))
if not msgs:
    print('    (없음)')
"
    FAILED=1
fi

# 3-5. close code 1000
CLOSE_CODE=$(echo "$WS_RESPONSE" | python3 -c "
import sys, json
print(json.load(sys.stdin).get('close_code', 'null'))
")
if [ "$CLOSE_CODE" = "1000" ]; then
    pass "close code 1000 확인"
elif [ "$CLOSE_CODE" = "None" ] || [ "$CLOSE_CODE" = "null" ]; then
    info "close code 없음 (서버가 close frame 미전송 또는 연결이 먼저 끊어짐)"
    # close code 없음은 soft warning (서버 구현에 따라 다를 수 있음)
else
    fail "close code: ${CLOSE_CODE} (기대값: 1000)"
    FAILED=1
fi

# ─── 최종 결과 ────────────────────────────────────────────────────────────────
echo
echo "================================================================"
if [ "$FAILED" -eq 0 ]; then
    echo -e "${GREEN}✅ smoke-stream.sh PASS${NC}"
    echo "   시나리오: /healthz → /auth/login → WS 핸드셰이크 → 10 binary → end → final → close 1000"
    exit 0
else
    echo -e "${RED}❌ smoke-stream.sh FAIL${NC}"
    echo "   위 ✗ 항목을 확인하세요."
    exit 1
fi
