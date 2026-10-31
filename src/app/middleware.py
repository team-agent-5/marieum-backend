"""
공통 미들웨어 — X-Trace-Id 자동 발급·echo, X-Server-Version 응답 헤더.

**순수 ASGI 미들웨어**로 작성. `BaseHTTPMiddleware`는 SSE/스트리밍 응답을
버퍼링해 끊어버리는 Starlette 알려진 이슈가 있어서 사용하지 않음.

api-spec.md §1.2 / §1.3 참조.
"""

from __future__ import annotations

import uuid

from starlette.datastructures import State
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class TraceIdMiddleware:
    """`X-Trace-Id`를 요청에서 꺼내거나 서버가 발급 → `scope.state`에 저장 → 응답에 echo."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 요청 헤더에서 추출 (ASGI headers는 list[tuple[bytes, bytes]])
        incoming = None
        for name, value in scope.get("headers", []):
            if name == b"x-trace-id":
                incoming = value.decode("latin-1")
                break
        trace_id = incoming or str(uuid.uuid4())

        # scope.state에 저장 (라우트 핸들러에서 Request.state.trace_id로 접근).
        # Starlette의 Request.state는 scope["state"]를 State 객체로 래핑하므로
        # 미들웨어에서도 State 인스턴스로 넣어줘야 함.
        if "state" not in scope or not isinstance(scope.get("state"), State):
            scope["state"] = State()
        scope["state"].trace_id = trace_id

        trace_header = (b"x-trace-id", trace_id.encode("latin-1"))

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # 중복 방지: 이미 있으면 교체
                headers = [(n, v) for n, v in headers if n != b"x-trace-id"]
                headers.append(trace_header)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


class ServerVersionMiddleware:
    def __init__(self, app: ASGIApp, version: str) -> None:
        self.app = app
        self._header = (b"x-server-version", version.encode("latin-1"))

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        header = self._header

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers = [(n, v) for n, v in headers if n != b"x-server-version"]
                headers.append(header)
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


def get_trace_id(request: Request) -> str:
    """라우트 핸들러·예외 핸들러에서 꺼내 쓰는 헬퍼.

    `TraceIdMiddleware`가 반드시 등록돼 있어야 함.
    """
    try:
        return request.state.trace_id
    except AttributeError:
        return str(uuid.uuid4())
