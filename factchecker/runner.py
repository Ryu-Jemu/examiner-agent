"""단일 백엔드 API: 텍스트 → 최종 리포트.

CLI·Gradio·평가 하니스가 모두 이 함수를 통해 그래프를 실행한다.
"""

from __future__ import annotations

import logging

from .config import get_settings
from .graph import compile_graph
from .models import FinalReport
from .state import FactCheckState

logger = logging.getLogger("factchecker.runner")

_COMPILED = None


def _get_compiled():
    """컴파일된 그래프 싱글턴(반복 실행 시 재컴파일 방지)."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = compile_graph()
    return _COMPILED


def run_factcheck_state(
    text: str, *, graph=None, recursion_limit: int | None = None
) -> FactCheckState:
    """그래프를 실행하고 최종 State 전체를 반환한다."""
    settings = get_settings()
    graph = graph or _get_compiled()
    rl = recursion_limit or (settings.max_loops * 6 + 10)
    initial: FactCheckState = {"input_text": text or ""}
    final_state = graph.invoke(initial, config={"recursion_limit": rl})
    return final_state


def run_factcheck(
    text: str, *, graph=None, recursion_limit: int | None = None
) -> FinalReport:
    """그래프를 실행하고 최종 리포트(FinalReport)를 반환한다."""
    final_state = run_factcheck_state(
        text, graph=graph, recursion_limit=recursion_limit
    )
    report = final_state.get("final_report")
    if report is None:
        # 그래프가 리포트를 만들지 못한 극단적 상황의 안전 폴백
        from .models import VerdictLabel

        report = FinalReport(
            overall_grade=VerdictLabel.INSUFFICIENT,
            overall_confidence=0.0,
            claim_breakdown=[],
            technique_tags=final_state.get("technique_tags", []) or [],
            rebuttal_card="검증 결과를 생성하지 못했습니다. 입력을 확인해 주세요.",
        )
    return report
