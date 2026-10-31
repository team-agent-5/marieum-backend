"""
ko-sroberta 임베딩 모델 싱글턴.
L2 의도분류 + RAG 검색 둘 다 공유해서 메모리 500MB를 한 번만 쓴다.
"""

from __future__ import annotations

import asyncio
import logging

from sentence_transformers import SentenceTransformer

from app.config import settings

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embedding_dim() -> int:
    return get_embedder().get_sentence_embedding_dimension() or 768


async def embed_text(text: str) -> list[float]:
    """Async wrapper over ko-sroberta encode. Uses asyncio.to_thread to prevent event-loop block.

    Returns 768-dim list[float]. Raises if embedder fails to load.
    """
    model = get_embedder()
    vec = await asyncio.to_thread(model.encode, text)
    return [float(x) for x in vec]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Async batch embed. Encodes all texts in a single to_thread call for efficiency.

    Returns list of 768-dim list[float], one per input text.
    """
    if not texts:
        return []
    model = get_embedder()
    vecs = await asyncio.to_thread(model.encode, texts)
    return [[float(x) for x in vec] for vec in vecs]


def parse_pgvector(v) -> list[float] | None:
    """pgvector 결과를 list[float]로 정규화.

    raw text() 쿼리로 vector 컬럼을 읽으면 psycopg3가 '[0.1,0.2,...]' 또는
    '{0.1,0.2,...}' 문자열을 반환하는 케이스가 있음. 파싱 실패 시 None 반환
    (예외 던지지 않음) — write-path 베스트에포트 원칙과 일관.
    """
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    if isinstance(v, str):
        s = v.strip().strip("[](){}").strip()
        if not s:
            return None
        try:
            return [float(x) for x in s.split(",")]
        except ValueError:
            return None
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        return None
