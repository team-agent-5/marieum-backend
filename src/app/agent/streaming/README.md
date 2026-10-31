# agent/streaming — 실시간 모드 LLM 교정 agent

## 역할

WebSocket(`/ws/phrase/stream`)을 통해 ML ASR 중간 결과를 수신하고, LLM으로 교정한 뒤 앱에 push한다.

| 모듈 | 역할 |
|---|---|
| `schemas.py` | WebSocket 메시지 전용 DTO (HTTP `app.schemas` 와 분리) |
| `router.py` | WebSocket 라우터 (`/ws/phrase/stream`) |
| `session.py` | 세션 state 관리 (handshake → 오디오 수신 → 교정 → 종료) |
| `pipeline.py` | LLM 교정 파이프라인 (debounce 150ms + version + fallback) |
| `ml_proxy.py` | ML ASR WebSocket 클라이언트 프록시 |

## 격리 조건

| 방향 | 허용 여부 |
|---|---|
| `agent/streaming/*` → `agent/ranking/*` import | **금지** |
| `agent/ranking/*` → `agent/streaming/*` import | **금지** |
| `agent/streaming/*` → `app/services/*` import | 허용 |
| `agent/streaming/*` → `app/common/*` import | 허용 (추후 생성 시) |

ruff `tidy-imports`(TID251) 설정이 CI에서 상호 import를 거부한다. `pyproject.toml` `[tool.ruff.lint.flake8-tidy-imports.banned-api]` 참조.

## 주요 결정 (시니어 리드 §2.1)

- `phrase_log` INSERT는 `/phrase/bypass` HTTP 라우트 담당. **이 패키지가 DB에 직접 쓰지 않는다.**
- JWT는 `?token=<jwt>` 쿼리 파라미터로 수신 (Cloudflare/브라우저 호환).
- `trace_id`는 서버가 WebSocket accept 직후 UUIDv4 발급. 클라이언트 입력 무시.
- `version_id`는 session layer에서 strip. 앱에는 `type/text/final/trace_id/confidence` 5필드만 노출.
- LLM 교정 비활성 토글: 환경변수 `MARIEUM_STREAMING_LLM_ENABLED=0`.
