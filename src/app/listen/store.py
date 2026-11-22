"""Listen 세션 메모리 store — 단일 uvicorn 프로세스 전제.

책임:
- 환자당 1개 ListenSession 유지
- join_code ↔ session, listener_token ↔ session 양방향 조회
- 상대방 WebSocket 핸들 보관 (중계용)
- utterance_id 멱등 (최근 50개 캐시)
- 30초 재접속 grace
- 만료 정리 (background task)
- join 시도 rate limit (IP당 분당 10회)

격리:
- `agent/`를 import하지 않는다.
- streaming/session.py가 이 모듈을 import하는 건 허용 (ranking이 아니므로 격리 룰 유지).

구현 Agent 참고:
- 모든 변경은 asyncio.Lock으로 보호 (단일 이벤트 루프라도 await 경계에서 재진입 가능).
- WebSocket 타입은 fastapi.WebSocket. send_json 호출은 상대방 쪽에서만.
- 만료/grace 정리는 10초 주기 background task로 처리.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from fastapi import WebSocket

# ────────── 상수 ──────────

JOIN_CODE_LEN = 6
JOIN_CODE_CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"  # 0/O, 1/I 제외 (Crockford-style).
JOIN_TTL = timedelta(minutes=10)      # join_code 유효기간
SESSION_TTL = timedelta(hours=2)      # 세션 전체 수명
RECONNECT_GRACE_S = 30                # 상대방 WS 끊긴 뒤 재접속 유예
UTTERANCE_DEDUP_WINDOW = 50           # 최근 utterance_id 캐시 크기
LISTENER_TOKEN_BYTES = 32             # secrets.token_urlsafe() 바이트 수

# join_code 충돌 체크 시 재시도 상한 (실질 충돌 확률은 32^6 / 환자 수 ≪ 1).
MAX_JOIN_CODE_RETRIES = 8

# IP당 /listen/join 분당 허용 횟수.
JOIN_RATE_LIMIT_PER_MIN = 10


# ────────── 데이터 ──────────


@dataclass
class ListenSession:
    """환자 1명당 1건."""

    listen_session_id: str                         # "ls_<ulid>"
    patient_id: UUID
    join_code: str                                 # 6자. 소비되면 다른 값으로 교체되지 않고 그대로 유지(1회용 정책).
    join_code_consumed: bool = False               # True면 신규 join 거절.
    listener_token: str | None = None              # join 성공 시 발급. 세션 1회만.
    join_expires_at: datetime = field(default_factory=lambda: datetime.now().astimezone() + JOIN_TTL)
    expires_at: datetime = field(default_factory=lambda: datetime.now().astimezone() + SESSION_TTL)
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    # 런타임 상태 (persist 안 됨)
    listener_ws: WebSocket | None = None           # 활성 상대방 소켓. None이면 미연결/grace 중.
    grace_until: datetime | None = None            # listener_ws None이면서 grace 중이면 설정됨.
    ended: bool = False
    end_reason: str | None = None                  # "patient_left" | "expired" | "revoked"

    # utterance_id 멱등 캐시. 최근 N개 (OrderedDict처럼 삽입순 유지).
    recent_utterances: dict[str, bool] = field(default_factory=dict)


# ────────── 예외 ──────────


class JoinCodeInvalidError(Exception):
    """코드가 없거나 만료/소비됨 → 404 LISTEN_CODE_INVALID."""


class ListenerAlreadyJoinedError(Exception):
    """해당 세션에 이미 다른 상대방이 연결/grace 중 → 409 LISTENER_ALREADY_JOINED."""


class RateLimitedError(Exception):
    """IP 분당 시도 한도 초과 → 429 RATE_LIMITED."""


# ────────── 공개 API — 구현 Agent가 채운다 ──────────


_lock: asyncio.Lock  # 모듈 전역 락. 초기화는 구현부에서.


def create_session(patient_id: UUID) -> ListenSession:
    """환자용 세션을 새로 만든다.

    - 같은 환자의 기존 세션이 있으면 `end_session(..., "revoked")`로 종료 후 교체.
    - 교체 시 기존 세션의 listener_ws에게 `WSListenSessionEndOut(reason="revoked")` 전송 후 close(4004).
    - 신규 join_code 발급 (충돌 시 재시도).
    """
    raise NotImplementedError


def get_by_patient(patient_id: UUID) -> ListenSession | None:
    """환자의 현재 세션. 없거나 ended이면 None."""
    raise NotImplementedError


def get_by_join_code(code: str) -> ListenSession | None:
    """코드로 조회. 소비됐거나 join 만료됐으면 None (혹은 반환하되 호출자가 체크)."""
    raise NotImplementedError


def get_by_listener_token(token: str) -> ListenSession | None:
    """토큰으로 조회. ended면 None."""
    raise NotImplementedError


async def consume_join_code(code: str, ip: str) -> tuple[ListenSession, str]:
    """`/listen/join` 메인 로직. 코드 → listener_token 교환.

    정상:
        - join_code_consumed=True 세팅, listener_token 발급 후 반환.
    예외:
        - JoinCodeInvalidError: 코드 없음 / 소비됨 / join_expires_at 경과.
        - ListenerAlreadyJoinedError: 이미 연결됨 OR grace 중.
        - RateLimitedError: IP 분당 10회 초과.

    rate limit은 **실패 시도**에 대해서만 카운트하는 게 아니라 **모든 시도** 카운트
    (브루트포스 대응). IP별 슬라이딩 1분 윈도우.
    """
    raise NotImplementedError


async def attach_listener(token: str, ws: WebSocket) -> ListenSession:
    """상대방 WS 연결 시 세션에 바인딩.

    - 세션 없거나 ended → close code 4004 유발 (예외 발생).
    - 이미 다른 listener_ws가 활성이면 close code 4009 (예외 발생).
    - grace 중이면 grace 해제 후 재바인딩 (listener:joined 재통지 안 함 — 내부 리커넥).
    """
    raise NotImplementedError


async def detach_listener(session: ListenSession, now_listener_left: bool = False) -> None:
    """상대방 WS 끊김.

    - now_listener_left=True면 즉시 "left" 상태로 전환 (grace 생략).
    - False면 RECONNECT_GRACE_S 초 동안 재접속 허용. 만료 시 "left" 전이.
    """
    raise NotImplementedError


async def end_session(session: ListenSession, reason: str) -> None:
    """세션 종료. 멱등.

    - ended=True, end_reason 세팅.
    - listener_ws가 있으면 WSListenSessionEndOut 전송 + close(4004).
    - 인덱스에서 제거는 cleanup task가 처리 (즉시 삭제도 가능).
    """
    raise NotImplementedError


def record_utterance(session: ListenSession, utterance_id: str, delivered: bool) -> None:
    """utterance 결과를 멱등 캐시에 기록.

    - 캐시 크기 UTTERANCE_DEDUP_WINDOW 초과 시 가장 오래된 항목 제거.
    - 이미 있으면 덮어쓰지 않는다 (첫 결과 유지).
    """
    raise NotImplementedError


def lookup_utterance(session: ListenSession, utterance_id: str) -> bool | None:
    """멱등 조회. 캐시에 있으면 저장된 delivered 반환, 없으면 None."""
    raise NotImplementedError


def is_listener_active(session: ListenSession) -> bool:
    """상대방이 실제로 소켓을 잡고 있나? grace 중이면 False.

    `say` 처리에서 `delivered` 판정에 사용.
    """
    raise NotImplementedError


# ────────── Background cleanup ──────────


async def start_cleanup_task() -> asyncio.Task:
    """주기적으로 만료 세션/grace 만료를 정리하는 백그라운드 태스크를 띄운다.

    - 10초 주기 권장.
    - grace_until이 지난 세션: listener:left 상태로 전이 (환자 WS에 통지는 streaming/session.py 담당).
    - expires_at이 지난 세션: end_session(session, "expired") 호출.
    - 반환된 태스크는 lifespan 종료 시 cancel 되어야 함.

    구현 메모:
    - 환자 WS로 listener 상태 변경 통지는 콜백 주입 방식 또는 store가 큐/이벤트로 노출.
      구현 Agent가 가장 간단한 방식 선택 (예: 세션에 asyncio.Queue를 두어 streaming이 소비).
    """
    raise NotImplementedError


# ────────── 환자↔상대방 중계 지원 ──────────


async def relay_say_to_listener(session: ListenSession, utterance_id: str, text: str) -> bool:
    """환자 app의 WSSayIn을 상대방에게 WSListenSayOut으로 전달.

    반환:
        True — 상대방 WS에 성공적으로 send.
        False — 상대방 WS가 없거나 send 실패 (grace, ended, close race 등).

    주의:
        - server_ts는 이 함수가 결정 (datetime.now().astimezone().isoformat()).
        - send 실패는 조용히 False 반환 (상대방 측 cleanup은 WS 수신 루프가 처리).
    """
    raise NotImplementedError


# ────────── Listener 상태 이벤트 (환자 WS로 전파용) ──────────


async def subscribe_listener_events(patient_id: UUID) -> asyncio.Queue[str]:
    """환자 WS가 listen 상태 변화를 수신하기 위한 큐를 받는다.

    큐 메시지는 ListenerState 문자열 ("joined" | "left" | "none").
    환자 WS가 끊기면 unsubscribe로 정리.

    구현 Agent 참고:
    - 환자 1명당 큐 1개로 충분 (환자 WS도 1개 cap).
    - 이미 subscribe돼 있으면 같은 큐 반환 or 새로 교체 — 구현자 재량.
    """
    raise NotImplementedError


async def unsubscribe_listener_events(patient_id: UUID) -> None:
    """환자 WS 종료 시 호출."""
    raise NotImplementedError
