"""judge.decide_route — 4가지 루프 종료 조건(순수 함수, LLM 불필요)."""

from factchecker.models import Verdict, VerdictLabel
from factchecker.nodes.judge import decide_route


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
