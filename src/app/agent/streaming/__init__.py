"""실시간 모드 LLM 교정 agent.

WebSocket을 통해 ASR 중간 결과를 수신하고 LLM으로 교정한 뒤 앱에 push한다.
각 모듈은 이 패키지 안에서만 완결된다.

하위 모듈:
- ``schemas``  : WebSocket 메시지 전용 DTO (WS 전용, app.schemas와 분리)
- ``router``   : WebSocket 라우터 (``/ws/phrase/stream``)
- ``session``  : 세션 state 관리
- ``pipeline`` : LLM 교정 파이프라인 (debounce + version + fallback)
- ``ml_proxy`` : ML ASR WebSocket 클라이언트 프록시

격리 규칙:
- ``app.agent.streaming.*`` → ``app.agent.ranking.*`` import 금지
- ruff tidy-imports(TID251)가 CI에서 거부한다.
- 공유 유틸(Groq 클라이언트 등)은 ``app.services.*`` 또는 ``app.common.*`` 를 통할 것.
"""
