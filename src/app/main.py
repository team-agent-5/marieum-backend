"""FastAPI 엔트리포인트 (v4 — 환자 1인 앱)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent.streaming.router import router as streaming_router
from app.api import auth, patient, phrase
from app.config import settings
from app.errors import install as install_error_handlers
from app.middleware import ServerVersionMiddleware, TraceIdMiddleware
from app.schemas import HealthzResponse

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Marieum Backend starting (env=%s)", settings.app_env)
    # 1. Preload ko-sroberta (prevents 10-15s cold load during first demo request)
    # NOTE: container must have the HF model pre-pulled to ~/.cache/huggingface/
    #       before first request; otherwise the warm() call here will trigger a download.
    from app.services.embedding import get_embedder
    embedder = get_embedder()
    _ = embedder.encode("물")  # warm
    log.info("Embedder preloaded: dim=%d", len(_))
    # 2. Cache phrase_catalog embeddings in-memory (small, ~25 rows)
    #    Rationale: avoids pgvector round-trip latency for F5 cold-start candidate pool.
    #    embedding stored as Python list — not pgvector internal type (concern #4).
    from sqlalchemy import select

    from app.db import AsyncSessionLocal
    from app.models import PhraseCatalog
    catalog: dict[int, dict] = {}
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(select(PhraseCatalog))).scalars().all()
        for r in rows:
            catalog[r.id] = {"text": r.text, "embedding": list(r.embedding) if r.embedding is not None else None}
    app.state.catalog_embeddings = catalog
    log.info("Catalog cache loaded: %d phrases", len(catalog))
    yield
    log.info("Shutting down.")
    try:
        from app.agent.llm_assist import _http_client
        if _http_client is not None:
            await _http_client.aclose()
    except Exception:
        pass


# ────────── FastAPI 앱 ──────────

tags_metadata = [
    {"name": "auth", "description": "환자 자율 가입·로그인 — JWT HS256, 365일 TTL"},
    {"name": "patient", "description": "환자 프로필·설정 (M2·S14·S15·S16)"},
    {"name": "phrase", "description": "**핵심** — 전사·추천·선택·preset·실시간·히스토리"},
    {"name": "streaming", "description": "실시간 대화 모드 (S16) — WebSocket /ws/phrase/stream"},
    {"name": "system", "description": "헬스체크·메타"},
]

APP_VERSION = "0.4.1"

app = FastAPI(
    title="Marieum Backend",
    description=(
        "## 구음장애 환자용 개인화 음성 AAC 앱 백엔드 (v4)\n\n"
        "- 환자 1인 앱 — 돌봄·관리자·보호자 UI 전면 제거\n"
        "- 추천 5축: 환자별 발화 이력·시간대·최근 선택·sequence·명시 선호 가중합\n"
        "- Privacy-first: 환자 본인 데이터만 사용 (개인정보보호법 §23)\n\n"
        "**통신 구조**\n"
        "- 환자앱 ↔ Backend: HTTP only. 녹음 오디오는 `POST /phrase/transcribe` (multipart/form-data)로 1회 전송.\n"
        "- Backend ↔ ML(ASR): HTTP (내부 proxy)\n\n"
        "자세한 명세: `docs/api-spec.md` (v4.1)\n"
        "기획서: `fastapi-roadmap-site/marieum-plan/dysarthria-care-aac-plan.md` (v4)"
    ),
    version=APP_VERSION,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    contact={"name": "Marieum Backend", "email": "backend@marieum.local"},
    license_info={"name": "Hackathon Internal"},
)

# ────────── Middleware ──────────

app.add_middleware(ServerVersionMiddleware, version=APP_VERSION)
app.add_middleware(TraceIdMiddleware)

# Cloudflare 통한 접근 가정. CORS는 환경변수로 좁힐 수 있도록.
_allowed_origins = (
    [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    if hasattr(settings, "allowed_origins") and settings.allowed_origins
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allowed_origins != ["*"],
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id", "X-Server-Version"],
)

# ────────── Error handlers ──────────

install_error_handlers(app)


# ────────── 라우터 등록 ──────────

app.include_router(auth.router)
app.include_router(patient.router)
app.include_router(phrase.router)
app.include_router(streaming_router)


# ────────── System ──────────


@app.get("/healthz", response_model=HealthzResponse, tags=["system"])
async def healthz() -> HealthzResponse:
    from app.agent.llm_assist import _circuit_open
    from app.config import settings

    if not settings.groq_api_key:
        groq_state = "down"  # disabled, no key
    elif _circuit_open():
        groq_state = "down"  # 429 circuit open
    else:
        groq_state = "ok"

    # 실제 라우터와 동일한 설정 사용. 설정 존재만으로 준비 완료를 단정하지 않는다.
    ml_asr_state = "unknown" if settings.ml_asr_ws_url else "down"

    return HealthzResponse(
        status="ok",
        version=app.version,
        ts=datetime.now(UTC),
        dependencies={
            "db": "unknown",
            "ml_asr": ml_asr_state,
            "groq_llm": groq_state,
            "anthropic": (
                "configured_proxy"
                if settings.anthropic_api_key
                and "copa.codyssey.kr" in settings.accuracy_anthropic_base_url
                else "configured"
                if settings.anthropic_api_key
                else "disabled"
            ),
        },
    )


@app.get("/health/warmup", tags=["system"])
async def warmup() -> dict:
    """Preload heavy models + sanity check. Call before demo."""
    import time

    from app.agent.llm_assist import _get_agent
    from app.services.embedding import get_embedder

    t0 = time.perf_counter()
    vec = get_embedder().encode("안녕하세요")
    t1 = time.perf_counter()
    catalog_count = len(getattr(app.state, "catalog_embeddings", {}))

    # Dry-run Agent construction (no network call) to warm the singleton
    agent = await _get_agent()
    agent_state = "constructed" if agent is not None else "disabled"

    return {
        "status": "ok",
        "embedder_dim": len(vec),
        "embed_latency_ms": int((t1 - t0) * 1000),
        "catalog_cached": catalog_count,
        "llm_agent": agent_state,
    }


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {
        "service": "marieum-backend",
        "version": app.version,
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
