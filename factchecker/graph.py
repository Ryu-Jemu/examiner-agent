"""LangGraph StateGraph 조립(전체 흐름은 README 참고)."""

from langgraph.graph import END, START, StateGraph

from .nodes import (
    adversarial_debate,
    extract_claims,
    judge,
    retrieve_evidence,
    route_after_judge,
    route_after_retrieve,
    synthesize,
    tag_techniques,
)
from .state import FactCheckState


def build_graph() -> StateGraph:
    """컴파일되지 않은 StateGraph 빌더를 반환한다."""
    builder = StateGraph(FactCheckState)

    builder.add_node("extract_claims", extract_claims)
    builder.add_node("retrieve_evidence", retrieve_evidence)
    builder.add_node("adversarial_debate", adversarial_debate)
    builder.add_node("judge", judge)
    builder.add_node("tag_techniques", tag_techniques)
    # defer=True: synthesize 는 병렬 분기(tag_techniques)와 루프 분기(judge)가 모두
    # 정착(settle)한 뒤 정확히 한 번 실행된다. fan-in + 루프 조기 발화 방지.
    builder.add_node("synthesize", synthesize, defer=True)

    builder.add_edge(START, "extract_claims")
    # 병렬 분기: 메인 체인 + 기법 태깅
    builder.add_edge("extract_claims", "retrieve_evidence")
    builder.add_edge("extract_claims", "tag_techniques")

    # 조건 분기: 재검색에서 신규 증거가 없으면 토론·판사 재실행은 입력이
    # 동일해 무의미 → 직전 판정으로 바로 합성(LLM 호출 절약). 첫 라운드는
    # 항상 토론으로 진행한다.
    builder.add_conditional_edges(
        "retrieve_evidence",
        route_after_retrieve,
        {
            "adversarial_debate": "adversarial_debate",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("adversarial_debate", "judge")

    # 조건 분기: 증거 부족 시 루프, 통과 시 합성
    builder.add_conditional_edges(
        "judge",
        route_after_judge,
        {"retrieve_evidence": "retrieve_evidence", "synthesize": "synthesize"},
    )

    # 랑데부: 병렬 분기와 메인 체인이 synthesize 에서 만남
    builder.add_edge("tag_techniques", "synthesize")
    builder.add_edge("synthesize", END)

    return builder


def compile_graph(checkpointer=None):
    """그래프를 컴파일한다. checkpointer 는 선택(기본 None, 결정론적 단발 실행)."""
    builder = build_graph()
    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
