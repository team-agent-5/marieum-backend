"""상대방 폰 듣기 (Listen) 패키지 — 환자 TTS 재생을 상대방 기기로 분리.

- 환자 폰 에코 피드백 제거가 목적.
- 단일 uvicorn 프로세스 전제 (worker 늘리면 그때 shared store로 교체).
- `agent/` import 금지 (streaming과 동일 격리 룰).
- 상대방 토큰은 JWT가 아닌 `secrets.token_urlsafe(32)` 랜덤 문자열 (권한 상승 방지).

설계: `docs/handoff/` 2026-11-22 (프론트 피드백 v0.3).
"""
