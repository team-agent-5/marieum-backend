#!/usr/bin/env bash
# Marieum Backend — Cloudflare Tunnel (quick tunnel) 관리 스크립트
#
# 서버에서 실행됨. uvicorn :8000 이 뜬 상태에서 호출.
# 로컬에서는 Makefile 의 `make tunnel-up / down / url / status / logs` 로 호출.
#
# cloudflared 없는 상태에서 처음 올리면 자동 설치 (sudo 필요).
# quick tunnel 이라 Cloudflare 계정/도메인 불필요. URL 은 trycloudflare.com 서브도메인.
# ⚠ 터널 재시작마다 URL 바뀜 → 서버 재배포(uvicorn 재기동)와 분리해서 운영.

set -euo pipefail

APP_PORT="${APP_PORT:-8000}"
STATE_DIR="${STATE_DIR:-$HOME/.marieum-tunnel}"
LOG_FILE="${STATE_DIR}/cloudflared.log"
URL_FILE="${STATE_DIR}/url.txt"
PID_FILE="${STATE_DIR}/pid"

mkdir -p "${STATE_DIR}"

ensure_cloudflared() {
  if command -v cloudflared >/dev/null 2>&1; then return; fi
  echo "→ cloudflared 미설치. 설치 시도 (sudo 필요)"
  arch="$(uname -m)"
  case "${arch}" in
    x86_64|amd64) deb="cloudflared-linux-amd64.deb" ;;
    aarch64|arm64) deb="cloudflared-linux-arm64.deb" ;;
    *) echo "✗ 지원 안 되는 아키텍처: ${arch}"; exit 1 ;;
  esac
  tmp="$(mktemp -d)"
  curl -fsSL -o "${tmp}/cloudflared.deb" \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/${deb}"
  sudo dpkg -i "${tmp}/cloudflared.deb" || sudo apt-get install -f -y
  rm -rf "${tmp}"
  echo "✓ cloudflared 설치 완료 ($(cloudflared --version | head -1))"
}

is_running() {
  [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

print_url_block() {
  local url="$1"
  cat <<EOF
🌐 Public URL (Cloudflare Tunnel)
   Base     : ${url}
   Swagger  : ${url}/docs
   ReDoc    : ${url}/redoc
   OpenAPI  : ${url}/openapi.json
   Health   : ${url}/healthz
EOF
}

cmd_up() {
  ensure_cloudflared
  if is_running; then
    echo "ℹ 이미 돌아가고 있음 (PID: $(cat "${PID_FILE}"))"
    cmd_url
    return
  fi
  : > "${LOG_FILE}"
  nohup cloudflared tunnel --no-autoupdate --protocol http2 --url "http://127.0.0.1:${APP_PORT}" \
    > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "→ cloudflared 시작 (PID: $(cat "${PID_FILE}"))"

  # URL 추출 대기 (최대 30초, 1초 간격)
  for _ in $(seq 1 30); do
    url=$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "${LOG_FILE}" | head -1 || true)
    if [ -n "${url}" ]; then
      echo "${url}" > "${URL_FILE}"
      echo ""
      print_url_block "${url}"
      return
    fi
    sleep 1
  done

  echo "✗ 30초 내에 URL 못 찾음. 로그 확인:"
  echo "   tail -f ${LOG_FILE}"
  exit 1
}

cmd_down() {
  if ! is_running; then
    echo "ℹ 안 돌고 있음"
    rm -f "${PID_FILE}" "${URL_FILE}"
    return
  fi
  kill "$(cat "${PID_FILE}")" || true
  rm -f "${PID_FILE}" "${URL_FILE}"
  echo "✓ 터널 종료"
}

cmd_url() {
  if [ ! -f "${URL_FILE}" ]; then
    echo "✗ URL 없음. 먼저: $0 up"
    exit 1
  fi
  print_url_block "$(cat "${URL_FILE}")"
}

cmd_status() {
  if is_running; then
    echo "✓ running (PID: $(cat "${PID_FILE}"))"
    [ -f "${URL_FILE}" ] && echo "  URL: $(cat "${URL_FILE}")"
  else
    echo "✗ stopped"
  fi
}

cmd_logs() {
  if [ ! -f "${LOG_FILE}" ]; then
    echo "✗ 로그 없음 (아직 up 안 했거나 state 디렉터리 비어있음)"
    exit 1
  fi
  tail -n 100 -f "${LOG_FILE}"
}

# ── systemd 서비스 설치 (재부팅 자동 시작 + 크래시 자동 재시작) ──
# sudo 1회 필요. 서비스 설치 후에는 다음과 같이 관리:
#   sudo systemctl status  marieum-tunnel
#   sudo systemctl restart marieum-tunnel
#   sudo systemctl stop    marieum-tunnel
#   journalctl -u marieum-tunnel -f
# URL 확인은 그대로 `tunnel.sh url` (상태 파일 재사용).
SERVICE_NAME="marieum-tunnel"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cmd_install_service() {
  ensure_cloudflared
  if [ ! -w /etc ] && ! sudo -n true 2>/dev/null; then
    echo "ℹ sudo 비번 1회 요구됨 (systemd unit 설치)"
  fi

  # 기존 nohup 터널 있으면 종료 (systemd 가 전담하도록)
  if is_running; then
    echo "→ 기존 nohup 터널 종료 (systemd 가 전담)"
    cmd_down
  fi

  USER_NAME="$(whoami)"
  CLOUDFLARED_BIN="$(command -v cloudflared)"
  HOME_DIR="${HOME}"

  sudo tee "${SERVICE_FILE}" > /dev/null <<UNIT
[Unit]
Description=Marieum Backend Cloudflare Tunnel (quick)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
Environment=HOME=${HOME_DIR}
ExecStartPre=/usr/bin/env bash -c 'mkdir -p ${STATE_DIR}'
ExecStart=${CLOUDFLARED_BIN} tunnel --no-autoupdate --protocol http2 --url http://127.0.0.1:${APP_PORT}
# cloudflared 가 로그를 stdout 으로 뱉으므로 journald 로 들어감.
# URL 추출용 파일 유지 위해, 별도 helper 로 journald 로그에서 URL 뽑아 저장.
ExecStartPost=/usr/bin/env bash -c 'for i in \$(seq 1 30); do url=\$(journalctl -u ${SERVICE_NAME} --since "60 seconds ago" --no-pager 2>/dev/null | grep -Eo "https://[a-z0-9-]+\\.trycloudflare\\.com" | tail -1); if [ -n "\$url" ]; then echo "\$url" > ${STATE_DIR}/url.txt; exit 0; fi; sleep 1; done; exit 0'
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT

  sudo systemctl daemon-reload
  sudo systemctl enable --now "${SERVICE_NAME}"

  echo "✓ systemd 서비스 설치/시작 완료: ${SERVICE_NAME}"
  echo ""
  echo "→ URL 추출 대기 (최대 30초)"
  for _ in $(seq 1 30); do
    if [ -f "${URL_FILE}" ] && [ -s "${URL_FILE}" ]; then
      echo ""
      print_url_block "$(cat "${URL_FILE}")"
      return
    fi
    sleep 1
  done
  echo "✗ URL 아직 못 받음. 상태 확인:"
  echo "   sudo systemctl status ${SERVICE_NAME}"
  echo "   journalctl -u ${SERVICE_NAME} -n 50"
}

cmd_uninstall_service() {
  if [ ! -f "${SERVICE_FILE}" ]; then
    echo "ℹ 서비스 설치돼 있지 않음"
    return
  fi
  sudo systemctl disable --now "${SERVICE_NAME}" || true
  sudo rm -f "${SERVICE_FILE}"
  sudo systemctl daemon-reload
  rm -f "${URL_FILE}"
  echo "✓ systemd 서비스 제거 완료"
}

cmd_service_status() {
  if [ ! -f "${SERVICE_FILE}" ]; then
    echo "ℹ systemd 서비스 미설치"
    return
  fi
  sudo systemctl status "${SERVICE_NAME}" --no-pager -n 10 || true
  echo ""
  [ -f "${URL_FILE}" ] && echo "URL: $(cat "${URL_FILE}")"
}

case "${1:-}" in
  up) cmd_up ;;
  down) cmd_down ;;
  url) cmd_url ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  restart) cmd_down; sleep 1; cmd_up ;;
  install-service) cmd_install_service ;;
  uninstall-service) cmd_uninstall_service ;;
  service-status) cmd_service_status ;;
  *) echo "사용법: $0 {up|down|restart|url|status|logs|install-service|uninstall-service|service-status}"; exit 1 ;;
esac
