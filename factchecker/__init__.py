"""단톡방 루머 적대적 팩트체커 + 미디어 리터러시 코치.

LangGraph 기반 멀티에이전트(검사·변호·판사·기법태거·반론작성) 팩트체커.
"""

from __future__ import annotations

__version__ = "0.1.0"


def run_factcheck(*args, **kwargs):
    """지연 임포트 래퍼 — `from factchecker import run_factcheck`.

    무거운 의존성(langgraph 등)을 패키지 임포트 시점이 아니라 실제 호출 시점에 로드한다.
    """
    from .runner import run_factcheck as _run

    return _run(*args, **kwargs)


def build_graph(*args, **kwargs):
    from .graph import build_graph as _build

    return _build(*args, **kwargs)


__all__ = ["__version__", "run_factcheck", "build_graph"]
