"""judge.decide_route — 4가지 루프 종료 조건(순수 함수, LLM 불필요).
retrieve 후 라우팅 — 재검색 신규 증거 0건이면 토론·판사 생략."""

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
        EvidenceItem(claim_id=0, snippet_id=f"ev_{i}", snippet="내용", source="테스트")
        for i in range(n)
    ]


def test_first_round_always_debates():
    state = {"loop_count": 0, "evidence_pool": [], "prev_pool_size": 0}
    assert route_after_retrieve(state) == "adversarial_debate"


def test_stalled_reretrieve_skips_to_synthesize():
    # 2라운드 재검색이 전부 중복 제거되어 풀 정체 → 토론·판사 생략
    state = {"loop_count": 1, "evidence_pool": _pool(3), "prev_pool_size": 3}
    assert route_after_retrieve(state) == "synthesize"


def test_reretrieve_with_new_evidence_debates():
    state = {"loop_count": 1, "evidence_pool": _pool(4), "prev_pool_size": 3}
    assert route_after_retrieve(state) == "adversarial_debate"
