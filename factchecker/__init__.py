"""Rumor Verification Agent. LangGraph 멀티에이전트 패키지."""

__version__ = "0.1.0"


def run_factcheck(*args, **kwargs):
    # 지연 임포트 래퍼.
    from .runner import run_factcheck as _run

    return _run(*args, **kwargs)


def build_graph(*args, **kwargs):
    from .graph import build_graph as _build

    return _build(*args, **kwargs)


__all__ = ["__version__", "run_factcheck", "build_graph"]
