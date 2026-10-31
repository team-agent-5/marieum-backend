"""
공통 에러 핸들러 — api-spec.md §1.4 ErrorResponse 포맷 통일.

HTTPException(detail=str) → ErrorResponse(error_code, message, trace_id, details?)
RequestValidationError    → ErrorResponse(error_code='VALIDATION_ERROR', details=[{field, issue}])
그 외 Exception           → ErrorResponse(error_code='INTERNAL', ...)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.middleware import get_trace_id

log = logging.getLogger(__name__)


# status code → api-spec.md §1.4 error_code 매핑. 핸들러에서 `detail`에 명시적으로
# {"error_code": "...", "message": "..."}를 넣어 보내면 그걸 우선 사용.
_STATUS_TO_CODE: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "VALIDATION_ERROR",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_409_CONFLICT: "CONFLICT",
    413: "AUDIO_TOO_LONG",
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: "INVALID_FORMAT",
    status.HTTP_503_SERVICE_UNAVAILABLE: "UPSTREAM_UNAVAILABLE",
    status.HTTP_504_GATEWAY_TIMEOUT: "TIMEOUT",
    status.HTTP_500_INTERNAL_SERVER_ERROR: "INTERNAL",
}


def _format(
    status_code: int,
    message: str,
    trace_id: str,
    details: list[dict[str, Any]] | None = None,
    error_code_override: str | None = None,
) -> JSONResponse:
    error_code = error_code_override or _STATUS_TO_CODE.get(status_code, "INTERNAL")
    body: dict[str, Any] = {
        "error_code": error_code,
        "message": message,
        "trace_id": trace_id,
    }
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def install(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException) -> JSONResponse:
        trace_id = get_trace_id(request)
        # detail이 dict이고 {error_code, message}를 포함하면 그대로 사용
        if isinstance(exc.detail, dict) and "error_code" in exc.detail:
            return _format(
                exc.status_code,
                exc.detail.get("message", ""),
                trace_id,
                details=exc.detail.get("details"),
                error_code_override=exc.detail["error_code"],
            )
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _format(exc.status_code, message, trace_id)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        trace_id = get_trace_id(request)
        details = [
            {"field": ".".join(str(x) for x in e["loc"][1:]), "issue": e["msg"]}
            for e in exc.errors()
        ]
        return _format(
            status.HTTP_400_BAD_REQUEST,
            "요청 형식이 올바르지 않습니다.",
            trace_id,
            details=details,
            error_code_override="VALIDATION_ERROR",
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        trace_id = get_trace_id(request)
        log.exception("Unhandled exception trace_id=%s", trace_id)
        return _format(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "서버 내부 오류가 발생했습니다.",
            trace_id,
        )
