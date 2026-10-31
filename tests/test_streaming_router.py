"""운영 주소 누락 시 Mock 없이 명시적으로 실패하는 WebSocket 계약."""

import uuid
from unittest.mock import patch

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from app.agent.streaming.router import router
from app.config import settings
from app.security import issue_token


def test_missing_ml_url_never_uses_mock(monkeypatch):
    monkeypatch.setattr(settings, "ml_asr_ws_url", "")
    app = FastAPI()
    app.include_router(router)
    token, _ = issue_token(uuid.uuid4(), "synthetic")
    with (
        patch("app.agent.streaming.session.MLProxySession") as proxy,
        TestClient(app) as client,
        client.websocket_connect("/ws/phrase/stream?token=" + token) as ws,
    ):
        error = ws.receive_json()
        assert error["error_code"] == "ML_UPSTREAM_UNAVAILABLE"
        ws.send_json({"type": "end"})
        ws.send_json({"type": "close"})
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == 1000
    proxy.assert_not_called()
