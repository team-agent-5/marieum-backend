# agent/ranking — 정확도 모드 ranking agent

> 2026-10-31 정리: 아래 재랭킹 설명은 의도된 설계이며 현재 ranker 호출 연결이 완료됐다는 뜻이 아닙니다. [두 모드 종합 설계](../../../../docs/agent-architecture.md) · [수정 실행 계획](../../../../docs/agent-mode-alignment-plan.md)을 확인하세요.


## 역할

환자의 발화 접두어(prefix) 및 컨텍스트를 기반으로 6-feature deterministic ranker + accuracy mode LLM rerank를 수행한다.

- 6-feature 점수: `prefix_match`, `patient_frequency`, `time_context`, `sequence_context`, `semantic_similarity`, `recency`
- accuracy mode: `AgentInput.accuracy_mode=True` 일 때 Groq LLM 재랭킹 활성화

> **현재 상태**: 뼈대(placeholder) 패키지. 실제 ranking 로직은 `app/agent/` 루트 (`ranker.py`, `pool.py`, `context.py`, `llm_assist.py`, `features/`)에 위치한다. 2차 작업 시 이곳으로 이관 예정.

## 격리 조건

| 방향 | 허용 여부 |
|---|---|
| `agent/ranking/*` → `agent/streaming/*` import | **금지** |
| `agent/streaming/*` → `agent/ranking/*` import | **금지** |
| `agent/ranking/*` → `app/services/*` import | 허용 |
| `agent/ranking/*` → `app/common/*` import | 허용 (추후 생성 시) |

ruff `tidy-imports`(TID251) 설정이 CI에서 상호 import를 거부한다. `pyproject.toml` `[tool.ruff.lint.flake8-tidy-imports.banned-api]` 참조.
