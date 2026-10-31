"""정확도 모드 ranking agent — 추후 구현.

현재 ranking 로직은 `app.agent` 루트(ranker.py, pool.py, context.py, …)에 위치한다.
이 패키지는 추후 해당 코드를 이곳으로 이관하기 위한 자리 확보(placeholder)다.

격리 규칙:
- ``app.agent.ranking.*`` → ``app.agent.streaming.*`` import 금지
- ruff tidy-imports(TID251)가 CI에서 거부한다.
"""
