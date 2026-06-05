"""route_after_judge / decide_route 4조건 종료 테스트."""

from __future__ import annotations

from factchecker.models import Verdict, VerdictLabel
from factchecker.nodes.judge import decide_route


def _v(needs: bool, conf: float = 0.5, cid: int = 0) -> Verdict:
    return Verdict(
        claim_id=cid,
        label=VerdictLabel.INSUFFICIENT,
        confidence=conf,
        needs_more_evidence=needs,
    )


def test_condition1_max_loops_caps():
    # loop_count 가 max_loops 에 도달 → synthesize
    out = decide_route(
        loop_count=2, verdicts=[_v(True)], pool_size=10, prev_pool_size=5,
        avg_conf=0.5, prev_conf=0.1, max_loops=2, threshold=0.05,
    )
    assert out == "synthesize"


def test_condition2_judge_satisfied():
    out = decide_route(
        loop_count=1, verdicts=[_v(False)], pool_size=10, prev_pool_size=5,
        avg_conf=0.9, prev_conf=0.1, max_loops=5, threshold=0.05,
    )
    assert out == "synthesize"


def test_condition3_no_new_evidence():
    # 풀이 자라지 않음 → 루프 무의미
    out = decide_route(
        loop_count=1, verdicts=[_v(True)], pool_size=8, prev_pool_size=8,
        avg_conf=0.5, prev_conf=0.1, max_loops=5, threshold=0.05,
    )
    assert out == "synthesize"


def test_condition4_confidence_converged():
    # 2회차 이상 + 신뢰도 변화 미미 → 진동 차단
    out = decide_route(
        loop_count=2, verdicts=[_v(True)], pool_size=12, prev_pool_size=8,
        avg_conf=0.50, prev_conf=0.49, max_loops=9, threshold=0.05,
    )
    assert out == "synthesize"


def test_loops_back_when_needed():
    # 추가 증거 필요 + 풀 성장 + 미수렴 + 캡 미도달 → 루프
    out = decide_route(
        loop_count=1, verdicts=[_v(True)], pool_size=12, prev_pool_size=8,
        avg_conf=0.8, prev_conf=0.1, max_loops=5, threshold=0.05,
    )
    assert out == "retrieve_evidence"


def test_empty_verdicts_synthesizes():
    out = decide_route(
        loop_count=0, verdicts=[], pool_size=0, prev_pool_size=-1,
        avg_conf=0.0, prev_conf=0.0, max_loops=2, threshold=0.05,
    )
    assert out == "synthesize"


def test_first_pass_not_stopped_by_convergence():
    # loop_count=1 에서는 수렴 조건(4)을 적용하지 않아 조기 종료하지 않음
    out = decide_route(
        loop_count=1, verdicts=[_v(True)], pool_size=12, prev_pool_size=8,
        avg_conf=0.0, prev_conf=0.0, max_loops=5, threshold=0.05,
    )
    assert out == "retrieve_evidence"


def test_route_after_judge_reads_decision():
    from factchecker.nodes.judge import route_after_judge

    assert route_after_judge({"route_decision": "synthesize"}) == "synthesize"
    assert route_after_judge({}) == "synthesize"  # 안전 기본값
