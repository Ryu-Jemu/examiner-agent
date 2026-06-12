"""judge.decide_route 4가지 루프 종료 조건(순수 함수, LLM 불필요).
retrieve 후 라우팅: 재검색 신규 증거 0건이면 토론·판사 생략."""

from factchecker.models import EvidenceItem, Verdict, VerdictLabel
from factchecker.nodes.judge import decide_route
from factchecker.nodes.retrieve_evidence import route_after_retrieve


def _verdict(needs_more: bool, confidence: float = 0.5) -> Verdict:
    return Verdict(
        claim_id=0,
        label=VerdictLabel.INSUFFICIENT,
        confidence=confidence,
        needs_more_evidence=needs_more,
    )


_BASE = dict(
    loop_count=1,
    pool_size=5,
    prev_pool_size=3,
    avg_conf=0.5,
    prev_conf=0.0,
    max_loops=2,
    threshold=0.05,
)


def test_loops_when_judge_wants_more_evidence():
    assert (
        decide_route(**_BASE, verdicts=[_verdict(True)])
        == "retrieve_evidence"
    )


def test_terminates_on_max_loops():
    args = {**_BASE, "loop_count": 2}
    assert decide_route(**args, verdicts=[_verdict(True)]) == "synthesize"


def test_terminates_when_judge_satisfied():
    assert decide_route(**_BASE, verdicts=[_verdict(False)]) == "synthesize"


def test_terminates_without_new_evidence():
    args = {**_BASE, "pool_size": 3, "prev_pool_size": 3}
    assert decide_route(**args, verdicts=[_verdict(True)]) == "synthesize"


def test_terminates_on_confidence_convergence():
    args = {**_BASE, "loop_count": 2, "max_loops": 4,
            "avg_conf": 0.51, "prev_conf": 0.50}
    assert decide_route(**args, verdicts=[_verdict(True)]) == "synthesize"


def test_terminates_without_verdicts():
    assert decide_route(**_BASE, verdicts=[]) == "synthesize"


def _pool(n: int) -> list[EvidenceItem]:
    return [
        EvidenceItem(claim_id=0, snippet_id=f"ev_{i}",
                     snippet="내용", source="테스트")
        for i in range(n)
    ]


def test_first_round_always_debates():
    state = {"loop_count": 0, "evidence_pool": [], "prev_pool_size": 0}
    assert route_after_retrieve(state) == "adversarial_debate"


def test_stalled_reretrieve_skips_to_synthesize():
    # 2라운드 재검색이 전부 중복 제거되어 풀이 정체되면 토론·판사를 생략한다
    state = {"loop_count": 1, "evidence_pool": _pool(3), "prev_pool_size": 3}
    assert route_after_retrieve(state) == "synthesize"


def test_reretrieve_with_new_evidence_debates():
    state = {"loop_count": 1, "evidence_pool": _pool(4), "prev_pool_size": 3}
    assert route_after_retrieve(state) == "adversarial_debate"


def test_stalled_reretrieve_emits_closing_turn(monkeypatch):
    """재검색 신규 0건이면 직전 판정으로 종합하는 마감 시스템 턴을 남긴다."""
    import importlib

    from factchecker.models import Claim, ClaimType

    # nodes/__init__ 이 동명 함수를 재수출하므로 모듈 객체를 직접 가져온다.
    re_mod = importlib.import_module("factchecker.nodes.retrieve_evidence")

    class _Doc:
        page_content = "내용"
        metadata = {"snippet_id": "ev_0", "source": "테스트",
                    "source_type": "학술"}

    class _Store:
        def similarity_search_with_relevance_scores(self, query, k):
            return [(_Doc(), 0.9)]  # 기존 풀과 동일 스니펫이라 전량 중복 제거

    import factchecker.rag.vectorstore as vs
    monkeypatch.setattr(vs, "get_or_build_evidence", lambda: _Store())

    claim = Claim(claim_id=0, text="주장",
                  claim_type=ClaimType.FACT, checkable=True)
    state = {
        "claims": [claim], "evidence_pool": _pool(1),  # ev_0 이미 풀에 존재
        "loop_count": 1,
    }
    update = re_mod.retrieve_evidence(state)
    assert update["evidence_pool"] == []
    assert update["retrieval_failed"] is False
    [turn] = update["debate_transcript"]
    assert turn.role == "판사" and "신규 증거가 없어" in turn.text

    # 첫 라운드(loop 0)에서는 마감 턴을 만들지 않는다
    state["loop_count"] = 0
    update = re_mod.retrieve_evidence(state)
    assert "debate_transcript" not in update
