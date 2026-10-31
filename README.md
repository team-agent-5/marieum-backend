# marieum-backend

**Marieum** — 구음장애(Dysarthria) 환자를 위한 음성 보조·추천 AAC 서비스. 이 레포는 **백엔드 서버**다.

## 서비스에서의 역할

Flutter 환자 앱을 지원하는 FastAPI 백엔드. 세 레이어로 구성:

### 1. 일반 API
회원가입·로그인(JWT), 환자 프로필·설정, 발화 히스토리, phrase 저장(write-time 임베딩 포함), preset 10개 처리 등 **환자 앱이 돌아가는 데 필요한 기본 CRUD·인증 라우트**.

### 2. ⚡ 실시간 LLM 교정 파이프라인
WebSocket `/ws/phrase/stream`. **Agent 아님** — ASR → Haiku 1-call 선형 파이프라인.

**구현**:
- 환자앱이 보낸 PCM16 100ms chunk를 ML ASR WebSocket으로 **pass-through 프록시**
- 수신한 interim transcript는 **150ms debounce + version-based stale drop**으로 묶음
- **Anthropic Claude Haiku 4.5** 한 번 호출로 "보수적 교정" (철자·띄어쓰기만, 과거/개인화 추측 금지)
- 교정 실패·타임아웃 시 **raw text fallback** (silent degradation)
- 환자앱이 받자마자 `flutter_tts`로 재생

### 3. ⭐ 정확도 추천 AI Agent
HTTP `/phrase/suggest`. **RAG-grounded LLM orchestrator** — 유일한 Agent 레이어.

**구현**:
- **PydanticAI 2.64** `Agent(..., output_type=X)` 로 Claude **Sonnet 4.5** 호출 (structured output 강제)
- `confidence_router`가 ASR 신뢰도로 **4-분기 결정**: 高/中/低/retry
- 각 분기별로 **RAG retrieval**: 환자 `phrase_log` + `phrase_catalog` + `patient_preference`를 **pgvector 코사인 + SQL**로 모아 LLM 컨텍스트 구성
- LLM이 환자 **과거 패턴·시간대·선호**를 종합해 추천 생성 (ML 신뢰도 낮은 低에선 ML 완전 무시)
- **3-level fallback**: LLM → 결정론 ranker(feature 가중합) → preset-10
- 서버측 **allowed-set 멤버십 체크**로 할루시 원천 봉쇄 (환자 코퍼스 외 phrase 거부)

**분기 정책**:
- 高 (≥0.98): 환자가 말한 그대로 재생 + **AI 선제 모달 1개**
- 中 (0.90–0.98): ML 1개 + **AI 추론 2개** = 총 3개 선택지
- 低 (0.40–0.90): ML 무시, **AI가 과거 데이터 기반 1개 추천**
- retry (<0.40): "다시 말씀해 주세요"

## 관련 레포

- [`marieum-app`](https://github.com/team-agent-5/marieum-app) — Flutter 환자 앱
- [`marieum-front`](https://github.com/team-agent-5/marieum-front) — 프론트
- [`marieum-ai`](https://github.com/team-agent-5/marieum-ai) — ASR ML 서버

## 스택

- Python 3.12 · FastAPI · SQLAlchemy 2.0 async
- Postgres 16 + pgvector 0.8 (IVFFlat cosine)
- Anthropic Claude (Haiku 4.5 교정 / Sonnet 4.5 추천) via PydanticAI
- ko-sroberta-multitask 임베딩 (768-dim)
- JWT (HS256), scrypt 비밀번호 해시

## 디렉토리 구조

```
src/app/
├── api/              # FastAPI 라우트 (일반 API)
│   ├── auth.py
│   ├── patient.py
│   └── phrase.py
├── agent/
│   ├── streaming/    # ⚡ 실시간 LLM 교정 파이프라인 (agent 아님)
│   │   ├── router.py             # WS /ws/phrase/stream
│   │   ├── session.py
│   │   ├── pipeline.py           # debounce + version stale drop
│   │   ├── ml_proxy.py           # ML ASR WebSocket 클라이언트
│   │   └── llm_corrector.py      # Haiku 교정 (1-call)
│   │
│   ├── features/     # ⭐ 정확도 Agent — 추천 feature 모듈
│   ├── branches/     #     ├ confidence 분기별 전략 (fallback)
│   ├── confidence_router.py      # ASR 신뢰도 → 분기 결정
│   ├── context.py    #     ├ 환자 context (phrase_log 집계)
│   ├── pool.py       #     ├ 후보 pool (phrase_log + catalog)
│   ├── llm_assist.py #     └ Sonnet PydanticAI Agent 호출
│   └── ranker.py     # 분기 라우팅 + 결과 종합
└── services/
    └── embedding.py  # ko-sroberta 벡터 생성 (write-time)
```

실시간 교정 레이어와 정확도 Agent 레이어는 **완전 격리** — 상호 import 금지 (ruff `tidy-imports` + 테스트로 강제).
