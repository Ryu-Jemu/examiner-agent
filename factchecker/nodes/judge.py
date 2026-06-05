"""Node 4: judge_and_self_refute (판사 + 자가 반박) + 루프 라우팅.

판사가 양측 논거와 근거 사슬을 종합해 5등급 + 보정 신뢰도를 산출하고, 이어 자가
반박(레드팀)으로 결론을 공격한다. 살아남지 못한 판정 중 증거가 빈약한 것은
needs_more_evidence=true 로 표시해 추가 검색 루프를 유도한다.

종료 판정(decide_route)은 4개의 독립 조건으로 무한루프·진동을 막는다.
"""

from __future__ import annotations

import logging

from .. import prompts
from ..config import get_settings
from ..llm import structured_invoke
from ..models import (
    Claim,
    EvidenceItem,
    RefutationList,
    Verdict,
    VerdictLabel,
    VerdictList,
)
from ..state import FactCheckState
from .formatting import (
    evidence_for_claim,
    format_arguments_block,
    format_claims_block,
    format_evidence_block,
    format_verdicts_block,
)

logger = logging.getLogger("factchecker.nodes.judge")

# 증거가 이 개수 이하이면 "빈약"으로 보고, 반론이 통과하지 못하면 추가 검색 유도
_THIN_EVIDENCE = 1


def _default_verdicts(claims: list[Claim]) -> list[Verdict]:
    """LLM 실패 시: 검증대상 주장을 모두 '불충분(판단 불가)'으로."""
    return [
        Verdict(
            claim_id=c.claim_id,
            label=VerdictLabel.INSUFFICIENT,
            confidence=0.0,
            evidence_chain=[],
            rationale="판정 생성에 실패하여 판단 불가로 처리했습니다.",
            needs_more_evidence=False,
        )
        for c in claims
        if c.checkable
    ]


def decide_route(
    *,
    loop_count: int,
    verdicts: list[Verdict],
    pool_size: int,
    prev_pool_size: int,
    avg_conf: float,
    prev_conf: float,
    max_loops: int,
    threshold: float,
) -> str:
    """4개 종료 조건. 'retrieve_evidence' 또는 'synthesize' 반환.

    (1) 최대 루프 도달  (2) 판사 만족(추가 증거 불필요)
    (3) 직전 라운드 대비 신규 증거 없음  (4) 신뢰도 수렴(진동 차단, 2회차부터)
    """
    if loop_count >= max_loops:               # (1) 하드 캡
        return "synthesize"
    if not verdicts:                          # 판정 없음 → 더 할 게 없음
        return "synthesize"
    if not any(v.needs_more_evidence for v in verdicts):  # (2) 판사 만족
        return "synthesize"
    if pool_size == prev_pool_size:           # (3) 신규 증거 없음 → 루프 무의미
        return "synthesize"
    if loop_count >= 2 and abs(avg_conf - prev_conf) < threshold:  # (4) 수렴
        return "synthesize"
    return "retrieve_evidence"


def judge(state: FactCheckState) -> dict:
    settings = get_settings()
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    arguments = state.get("arguments", []) or []
    loop = state.get("loop_count", 0)
    prev_conf = state.get("last_confidence", 0.0)
    prev_pool_size = state.get("prev_pool_size", -1)

    checkable = [c for c in claims if c.checkable]

    # --- 판정 ---
    if checkable:
        judge_prompt = prompts.render(
            "judge",
            claims_block=format_claims_block(claims),
            arguments_block=format_arguments_block(arguments),
            evidence_block=format_evidence_block(pool),
        )
        result = structured_invoke(
            judge_prompt, VerdictList, default=VerdictList(verdicts=_default_verdicts(claims))
        )
        verdicts = result.verdicts or _default_verdicts(claims)
        # 환각 가드: evidence_chain 은 실제 풀의 snippet_id 부분집합만 허용(검사/변호와 동일 방어).
        valid_ids = {e.snippet_id for e in pool}
        for v in verdicts:
            dropped = [s for s in v.evidence_chain if s not in valid_ids]
            if dropped:
                logger.warning("환각 evidence_chain 제거: %s (유효 id 아님)", dropped)
            v.evidence_chain = [s for s in v.evidence_chain if s in valid_ids]
    else:
        verdicts = []

    # --- 자가 반박(레드팀) ---
    refutations = []
    if verdicts:
        refute_prompt = prompts.render(
            "self_refute",
            verdicts_block=format_verdicts_block(verdicts),
            evidence_block=format_evidence_block(pool),
        )
        refute_result = structured_invoke(
            refute_prompt, RefutationList, default=RefutationList(refutations=[])
        )
        refutations = refute_result.refutations
        # loop 값 정규화
        for r in refutations:
            r.loop = loop

    # --- 자가 반박 결과를 루프 유도에 반영 ---
    not_survived = {r.claim_id for r in refutations if not r.survived}
    for v in verdicts:
        ev_count = len(evidence_for_claim(pool, v.claim_id))
        if v.claim_id in not_survived and ev_count <= _THIN_EVIDENCE:
            v.needs_more_evidence = True

    new_loop = loop + 1
    avg_conf = (
        sum(v.confidence for v in verdicts) / len(verdicts) if verdicts else 0.0
    )
    decision = decide_route(
        loop_count=new_loop,
        verdicts=verdicts,
        pool_size=len(pool),
        prev_pool_size=prev_pool_size,
        avg_conf=avg_conf,
        prev_conf=prev_conf,
        max_loops=settings.max_loops,
        threshold=settings.confidence_delta_threshold,
    )

    logger.info(
        "판정 %d건, 자가반박 %d건, loop=%d→%d, 라우팅=%s",
        len(verdicts), len(refutations), loop, new_loop, decision,
    )
    return {
        "verdicts": verdicts,                 # 교체(단일 쓰기)
        "refutation_log": refutations,        # add 리듀서
        "loop_count": new_loop,               # 교체(카운터)
        "last_confidence": avg_conf,          # 다음 라운드 비교 기준
        "route_decision": decision,           # 라우터가 읽음
    }


def route_after_judge(state: FactCheckState) -> str:
    """조건 분기 함수 — judge 가 계산한 결정을 그대로 사용한다."""
    return state.get("route_decision", "synthesize")
