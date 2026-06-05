"""State 리듀서 정확성 테스트(병렬 쓰기 gotcha 방지)."""

from __future__ import annotations

from operator import add
from typing import Annotated, get_args, get_type_hints

from typing_extensions import TypedDict


def _reducer_of(hints, field):
    meta = get_args(hints[field])
    return meta[1:] if len(meta) > 1 else ()


def test_accumulating_fields_use_add_reducer():
    from factchecker.state import FactCheckState

    hints = get_type_hints(FactCheckState, include_extras=True)
    for field in ("evidence_pool", "arguments", "refutation_log", "technique_tags"):
        assert add in _reducer_of(hints, field), f"{field} 는 add 리듀서를 가져야 함"


def test_single_writer_fields_have_no_reducer():
    from factchecker.state import FactCheckState

    hints = get_type_hints(FactCheckState, include_extras=True)
    # loop_count 는 카운터 → add 면 이중 카운트되므로 리듀서가 없어야 함
    assert add not in _reducer_of(hints, "loop_count")
    # verdicts/claims 는 단일 쓰기 → 교체
    assert add not in _reducer_of(hints, "verdicts")
    assert add not in _reducer_of(hints, "claims")


def test_parallel_writes_to_add_field_merge(monkeypatch):
    """add 리듀서가 있으면 병렬 쓰기가 InvalidUpdateError 없이 병합된다."""
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        items: Annotated[list, add]

    def a(_s):
        return {"items": ["a"]}

    def b(_s):
        return {"items": ["b"]}

    builder = StateGraph(S)
    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    graph = builder.compile()

    out = graph.invoke({"items": []})
    assert sorted(out["items"]) == ["a", "b"]
