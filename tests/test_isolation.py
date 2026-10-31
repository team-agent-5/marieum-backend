"""Agent 격리 조건 테스트.

ruff `banned-api`는 per-file-ignores (TID251)로 streaming·ranking 패키지 안에서
자기 패키지 import는 허용되지만, 그로 인해 ranking → streaming / streaming → ranking
상호 import도 ruff가 침묵하는 Loophole이 있음 (Phase B2 리뷰 매니저 D2 지적).

이 테스트는 ast 모듈로 import 문을 파싱해 상호 import를 **강제로 거부**함.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "app" / "agent"
STREAMING = SRC_ROOT / "streaming"
RANKING = SRC_ROOT / "ranking"


def _find_imports(path: Path) -> list[str]:
    """path 아래 모든 .py 파일에서 import 문자열 수집."""
    imports = []
    for py in path.rglob("*.py"):
        if py.name == "__pycache__":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"{py.relative_to(path.parent)}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(f"{py.relative_to(path.parent)}:{node.module}")
    return imports


def test_streaming_does_not_import_ranking() -> None:
    """agent/streaming/* 가 agent/ranking.* 를 import하지 않는지 AST로 확인."""
    if not STREAMING.exists():
        pytest.skip("streaming package not found")

    violations = [
        imp for imp in _find_imports(STREAMING)
        if "app.agent.ranking" in imp or ":agent.ranking" in imp
    ]
    assert not violations, (
        f"격리 조건 위반: streaming이 ranking을 import함:\n  "
        + "\n  ".join(violations)
        + "\n→ 공유 로직은 app.services 또는 app.common에 두세요."
    )


def test_ranking_does_not_import_streaming() -> None:
    """agent/ranking/* 가 agent/streaming.* 를 import하지 않는지 AST로 확인."""
    if not RANKING.exists():
        pytest.skip("ranking package not found")

    violations = [
        imp for imp in _find_imports(RANKING)
        if "app.agent.streaming" in imp or ":agent.streaming" in imp
    ]
    assert not violations, (
        f"격리 조건 위반: ranking이 streaming을 import함:\n  "
        + "\n  ".join(violations)
        + "\n→ 공유 로직은 app.services 또는 app.common에 두세요."
    )
