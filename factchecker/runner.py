"""단일 백엔드 API: 텍스트 → 최종 리포트(CLI·서버·평가가 모두 이 함수를 사용)."""

from .config import get_settings
from .graph import compile_graph
from .models import FinalReport
from .state import FactCheckState

_COMPILED = None


def _get_compiled():
    # 컴파일된 그래프 싱글턴(반복 실행 시 재컴파일 방지).
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = compile_graph()
    return _COMPILED


def run_factcheck_state(
    text: str,
    *,
    graph=None,
    recursion_limit: int | None = None,
    api_key: str | None = None,
    image_data_url: str | None = None,
) -> FactCheckState:
    """그래프를 실행해 최종 State 를 반환한다.

    api_key(BYOK)는 invoke 전에 요청 범위로 설정해야 병렬 노드 워커로 전파된다.
    image_data_url 은 첨부 이미지(data:image/...;base64,...) 멀티모달 입력.
    """
    from .llm import reset_request_api_key, set_request_api_key

    settings = get_settings()
    graph = graph or _get_compiled()
    rl = recursion_limit or (settings.max_loops * 6 + 10)
    initial: FactCheckState = {"input_text": text or ""}
    if image_data_url:
        initial["input_image"] = image_data_url

    token = set_request_api_key(api_key) if api_key else None
    try:
        return graph.invoke(initial, config={"recursion_limit": rl})
    finally:
        if token is not None:
            reset_request_api_key(token)


def report_from_state(final_state: FactCheckState) -> FinalReport:
    """최종 State 에서 리포트를 꺼낸다(없으면 안전 폴백)."""
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


def run_factcheck(
    text: str,
    *,
    graph=None,
    recursion_limit: int | None = None,
    api_key: str | None = None,
    image_data_url: str | None = None,
) -> FinalReport:
    """그래프를 실행해 최종 리포트(FinalReport)를 반환한다."""
    final_state = run_factcheck_state(
        text,
        graph=graph,
        recursion_limit=recursion_limit,
        api_key=api_key,
        image_data_url=image_data_url,
    )
    return report_from_state(final_state)
