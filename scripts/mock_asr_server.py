"""
Mock ML/ASR 서버. ML팀이 real /asr 올리기 전까지 백엔드가 선행 작업하기 위해.

실행:
    python scripts/mock_asr_server.py
    → http://localhost:9001/asr
"""

from __future__ import annotations

import asyncio
import random
import uuid

import uvicorn
from fastapi import FastAPI, Header, UploadFile

app = FastAPI()

SAMPLES = [
    {"text": "물 좀 주세요", "confidence": 0.87},
    {"text": "숨이 차요", "confidence": 0.72},
    {"text": "화장실 가고 싶어요", "confidence": 0.91},
    {"text": "아파요", "confidence": 0.65},
    {"text": "추워요", "confidence": 0.80},
    {"text": "약 주세요", "confidence": 0.78},
    {"text": "저기요", "confidence": 0.60},
    {"text": "넘어졌어요", "confidence": 0.88},
    {"text": "", "confidence": 0.05},  # 무음 시뮬
]


@app.get("/healthz")
async def health() -> dict:
    return {"status": "ok", "models": ["asr-mock"], "model_version": {"asr": "mock-v1"}}


@app.post("/asr")
async def asr(audio: UploadFile, x_trace_id: str | None = Header(default=None)) -> dict:
    await audio.read()
    await asyncio.sleep(0.4)
    r = random.choice(SAMPLES)
    return {
        **r,
        "duration_ms": 1200,
        "language": "ko",
        "model_version": "mock-v1",
        "no_speech_prob": 0.9 if r["text"] == "" else 0.03,
        "trace_id": x_trace_id or str(uuid.uuid4()),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
