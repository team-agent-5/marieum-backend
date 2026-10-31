from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_JWT_SECRET_DEV_DEFAULT = "dev-secret-change-me-in-production-32bytes-min"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # 서버
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # DB
    database_url: str = "postgresql+asyncpg://aac:aac@localhost:5432/aac"

    # ASR (HTTP mock 서버, Backend가 멀티파트로 받은 오디오 forward)
    ml_asr_url: str = "http://localhost:9001"
    ml_asr_timeout_sec: float = 3.0
    # ML ASR WebSocket URL (미설정 시 healthz에서 "down" 상태 반영)
    ml_asr_ws_url: str = Field(
        default="", validation_alias=AliasChoices("ML_ASR_WS_URL", "MARIEUM_ML_ASR_WS_URL")
    )

    # LLM (Groq)
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_timeout_sec: float = 1.5

    # 실시간 교정 (Anthropic Haiku): 현재 발화만 전달, 개인 문맥 미사용
    anthropic_api_key: str = Field(default="", repr=False)
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_timeout_sec: float = Field(default=1.5, gt=0)
    streaming_llm_enabled: bool = Field(default=True, validation_alias="MARIEUM_STREAMING_LLM_ENABLED")

    # Anthropic (Claude) — accuracy mode agent
    # Reuses ANTHROPIC_API_KEY for unified key management; copa proxy for consistent routing.
    # NOTE: 'claude-sonnet-4-5' alias는 프록시 라우팅 실패 가능.
    # 운영자가 `curl -s https://copa.codyssey.kr/v1/models | jq '.data[].id'`
    # 로 proxy가 받는 model ID 리스트 확인 필수.
    accuracy_anthropic_model: str = "claude-sonnet-4-5-20250929"  # dated snapshot (safer than alias)
    accuracy_anthropic_base_url: str = "https://copa.codyssey.kr/v1"  # proxy (unified with streaming)
    accuracy_anthropic_timeout_sec: float = Field(default=1.5, gt=0)

    # Embedding
    embedding_model: str = "jhgan/ko-sroberta-multitask"

    # Guardrails
    intent_confidence_threshold: float = 0.60
    retry_max_attempts: int = 3

    # Agent — Confidence 분기 임계값.
    # 운영 중 튜닝 가능 — ML 분포 변동 시 env 로 override.
    retry_confidence: float = 0.40  # env: RETRY_CONFIDENCE — 재질의 상한
    low_upper: float = 0.90         # env: LOW_UPPER        — 저신뢰 상한 (중신뢰 시작)
    mid_upper: float = 0.98         # env: MID_UPPER        — 중신뢰 상한 (고신뢰 시작)

    # JWT (HS256, 리프레시 없음)
    jwt_secret: str = _JWT_SECRET_DEV_DEFAULT
    jwt_algorithm: str = "HS256"

    # CORS (Cloudflare 통한 접근 등. 쉼표 구분, 빈 값이면 "*")
    allowed_origins: str = ""

    @model_validator(mode="after")
    def _enforce_prod_secret(self) -> "Settings":
        # prod/staging 환경에서는 기본 JWT 시크릿을 그대로 두면 기동 거부.
        if self.app_env not in ("dev", "test") and self.jwt_secret in ("", _JWT_SECRET_DEV_DEFAULT):
            raise ValueError(
                f"JWT_SECRET must be overridden in app_env={self.app_env!r}. "
                "Set JWT_SECRET env var to a 32+ byte random string."
            )
        if len(self.jwt_secret) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters.")
        return self


settings = Settings()
