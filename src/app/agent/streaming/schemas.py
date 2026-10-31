"""실시간 모드 streaming agent WebSocket 메시지·에러 스키마.

메인 `app.schemas`를 재export하는 네임스페이스. 지금은 모든 streaming 전용 심볼이
`app.schemas`에 정의돼 있고, 이 모듈은 '위치 지시자' 역할만 한다.

**격리 조건**:
- `agent/streaming/`에서만 사용. ranking agent는 이 모듈을 import하지 않는다.
- 반대로 ranking agent 전용 심볼 (Candidate, SuggestResponse 등)은 이 모듈에서 재export하지 않는다.

**재export 심볼** (전부 `app.schemas`에서):
- WSAudioIn, WSEndIn, WSCancelIn, WSClientMessage
- WSTranscriptOut, WSErrorOut, WSEndNoticeOut, WSServerMessage
- WSErrorCode, WSCloseCode

**streaming 전용 추가 심볼** (이 모듈에서 정의):
- 지금은 없음. 추가 필요하면 app.schemas 가 아니라 **여기에** 정의하고, 메인 schemas 로는 올리지 않는다.
"""

from __future__ import annotations

from app.schemas import (
    WSAudioIn,
    WSCancelIn,
    WSClientMessage,
    WSCloseCode,
    WSEndIn,
    WSEndNoticeOut,
    WSErrorCode,
    WSErrorOut,
    WSServerMessage,
    WSTranscriptOut,
)

__all__ = [
    "WSAudioIn",
    "WSCancelIn",
    "WSClientMessage",
    "WSCloseCode",
    "WSEndIn",
    "WSEndNoticeOut",
    "WSErrorCode",
    "WSErrorOut",
    "WSServerMessage",
    "WSTranscriptOut",
]
