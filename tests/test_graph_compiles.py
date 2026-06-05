"""P0 DoD: 그래프가 LLM 호출 없이 컴파일된다."""

from __future__ import annotations


def test_build_graph_compiles():
    from factchecker.graph import build_graph

    compiled = build_graph().compile()
    assert compiled is not None


def test_compile_graph_helper():
    from factchecker.graph import compile_graph

    assert compile_graph() is not None


def test_graph_nodes_present():
    from factchecker.graph import build_graph

    builder = build_graph()
    # 6개 노드가 모두 등록되어야 함
    expected = {
        "extract_claims",
        "retrieve_evidence",
        "adversarial_debate",
        "judge",
        "tag_techniques",
        "synthesize",
    }
    assert expected.issubset(set(builder.nodes.keys()))
