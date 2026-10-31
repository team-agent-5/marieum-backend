.PHONY: install db-up db-down migrate upgrade dev mock-asr seed ingest test fmt \
        deploy tunnel-up tunnel-down tunnel-restart tunnel-url tunnel-status tunnel-logs \
        tunnel-install tunnel-uninstall tunnel-service-status

# ── NCP 배포/터널 공용 변수 (deploy.sh 와 동일) ──────────────────────────
HOST       ?= 10.1.1.11
SSH_USER   ?= ncloud
SSH_PORT   ?= 22
SSH_KEY    ?= $(HOME)/.ssh/marieum_ncp
REMOTE_DIR ?= /home1/$(SSH_USER)/marieum-backend
SSH        = ssh -i $(SSH_KEY) -p $(SSH_PORT) -o IdentitiesOnly=yes \
             -o StrictHostKeyChecking=accept-new $(SSH_USER)@$(HOST)

install:
	pip install -e ".[dev,rag-ingest]"

db-up:
	docker compose up -d

db-down:
	docker compose down

# Alembic revision 생성 (예: make migrate m="add patients table")
migrate:
	alembic revision --autogenerate -m "$(m)"

upgrade:
	alembic upgrade head

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --app-dir src

mock-asr:
	python scripts/mock_asr_server.py

seed:
	python scripts/seed.py

ingest:
	python scripts/ingest_guides.py

test:
	pytest -v

fmt:
	ruff check --fix src tests
	ruff format src tests

# ── 배포 (NCP) ─────────────────────────────────────────────────────────
deploy:
	bash scripts/deploy.sh

# ── Cloudflare Tunnel (외부 공개용) ────────────────────────────────────
# quick tunnel = 익명/무료/무제한. URL 은 재시작마다 바뀌므로 가급적 up 유지.
# 서버에 cloudflared 없으면 tunnel.sh 가 자동 설치 (sudo).
tunnel-up:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh up"

tunnel-down:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh down"

tunnel-restart:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh restart"

tunnel-url:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh url"

tunnel-status:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh status"

tunnel-logs:
	$(SSH) "bash $(REMOTE_DIR)/scripts/tunnel.sh logs"

# ── systemd 서비스 (재부팅 자동 시작 + 크래시 자동 재시작) ──
# install 하면 nohup 터널은 종료되고 systemd 가 전담. URL 확인은 그대로 tunnel-url.
# sudo 비번 1회 요구됨.
tunnel-install:
	$(SSH) -t "bash $(REMOTE_DIR)/scripts/tunnel.sh install-service"

tunnel-uninstall:
	$(SSH) -t "bash $(REMOTE_DIR)/scripts/tunnel.sh uninstall-service"

tunnel-service-status:
	$(SSH) -t "bash $(REMOTE_DIR)/scripts/tunnel.sh service-status"
