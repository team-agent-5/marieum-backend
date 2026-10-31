"""pytest 공통 fixture.

- 세션 시작 전 DB 재시드 (앞선 테스트 흔적 제거).
- httpx AsyncClient + ASGITransport로 라이브 FastAPI 앱 호출.
- 로그인 헬퍼: 환자·돌봄 간호사·돌봄 보호사·관리자 토큰을 session-scope로 캐싱.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app

PATIENT_PIN = "4321"
CAREGIVER_PIN = "1234"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "marieum2026"


def _reseed_db() -> None:
    """세션 시작 시 1회 — DB truncate + seed."""
    backend_root = Path(__file__).resolve().parent.parent
    subprocess.run(
        [
            "docker",
            "exec",
            "aac-db",
            "psql",
            "-U",
            "aac",
            "-d",
            "aac",
            "-c",
            "TRUNCATE TABLE call_responses, feedback, calls, audit_log, documents, patients, caregivers RESTART IDENTITY CASCADE;",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "scripts/seed.py"],
        cwd=backend_root,
        env={"PYTHONPATH": str(backend_root / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        check=True,
        capture_output=True,
    )


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _reseed_before_session():
    _reseed_db()
    yield


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ────────── 로그인 헬퍼 (session-scope 캐시로 반복 로그인 방지) ──────────


@pytest_asyncio.fixture(scope="session")
async def tokens() -> dict[str, str]:
    """세션 전체에서 공유하는 토큰 세트. {patient, nurse, caregiver, admin, nurse_id, caregiver_id, patient_id}."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        # admin
        r = await c.post(
            "/auth/admin/login",
            json={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
        )
        admin_token = r.json()["access_token"]

        # patient (seed의 첫 환자)
        r = await c.post(
            "/auth/patients/login",
            json={"patient_code": "demo-patient-code", "pin": PATIENT_PIN},
        )
        patient_token = r.json()["access_token"]
        patient_id = r.json()["subject"]["id"]

        # caregivers public → nurse/caregiver UUID 찾기
        r = await c.get("/caregivers/public")
        rows = r.json()
        nurse_id = next(c["id"] for c in rows if c["role"] == "nurse")
        helper_id = next(c["id"] for c in rows if c["role"] == "caregiver")

        r = await c.post(
            "/auth/caregivers/login",
            json={"caregiver_id": nurse_id, "pin": CAREGIVER_PIN},
        )
        nurse_token = r.json()["access_token"]

        r = await c.post(
            "/auth/caregivers/login",
            json={"caregiver_id": helper_id, "pin": CAREGIVER_PIN},
        )
        helper_token = r.json()["access_token"]

    return {
        "patient": patient_token,
        "nurse": nurse_token,
        "helper": helper_token,
        "admin": admin_token,
        "nurse_id": nurse_id,
        "helper_id": helper_id,
        "patient_id": patient_id,
    }


def hdr(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}
