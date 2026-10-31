"""실제 ML 환경변수 이름과 기존 배포 설정의 호환성."""

import pytest

from app.config import Settings


@pytest.mark.parametrize("name", ["ML_ASR_WS_URL", "MARIEUM_ML_ASR_WS_URL"])
def test_ml_url_accepts_both_environment_names(monkeypatch, name):
    monkeypatch.delenv("ML_ASR_WS_URL", raising=False)
    monkeypatch.delenv("MARIEUM_ML_ASR_WS_URL", raising=False)
    monkeypatch.setenv(name, "ws://10.1.2.11:8001/asr-stream")
    assert Settings(_env_file=None).ml_asr_ws_url == "ws://10.1.2.11:8001/asr-stream"


def test_canonical_ml_url_wins_over_legacy_alias(monkeypatch):
    monkeypatch.setenv("ML_ASR_WS_URL", "ws://10.1.2.11:8001/asr-stream")
    monkeypatch.setenv("MARIEUM_ML_ASR_WS_URL", "ws://127.0.0.1:19001/asr-stream")
    assert Settings(_env_file=None).ml_asr_ws_url == "ws://10.1.2.11:8001/asr-stream"
