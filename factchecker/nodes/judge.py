"""판사 노드: 양측 논거를 종합해 5등급 판정 + 자가 반박을 내고, 루프 종료를 결정한다."""

import logging

from .. import prompts
from ..config import get_settings
from ..llm import structured_invoke
from ..models import (
    POLAR_LABELS,
    Claim,
    DebateTurn,
    EvidenceItem,
    RefutationEntry,
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
)

logger = logging.getLogger("factchecker.nodes.judge")

# 증거가 이 개수 이하이면 빈약으로 보고 추가 검색 유도
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
    """종료 조건 중 하나라도 참이면 synthesize, 아니면 retrieve_evidence."""
    if loop_count >= max_loops:               # 하드 캡
        return "synthesize"
    if not verdicts:                          # 판정 없음
        return "synthesize"
    if not any(v.needs_more_evidence for v in verdicts):  # 판사 만족
        return "synthesize"
    if pool_size == prev_pool_size:           # 신규 증거 없음
        return "synthesize"
    if loop_count >= 2 and abs(avg_conf - prev_conf) < threshold:  # 수렴
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

    # 판정과 자가 반박을 한 번의 LLM 호출로 동시 산출
    if checkable:
        judge_prompt = prompts.render(
            "judge",
            claims_block=format_claims_block(claims),
            arguments_block=format_arguments_block(arguments),
            evidence_block=format_evidence_block(pool),
        )
        result = structured_invoke(
            judge_prompt,
            VerdictList,
            default=VerdictList(verdicts=_default_verdicts(claims)),
        )
        verdicts = result.verdicts or _default_verdicts(claims)
        # 환각 가드: evidence_chain 은 해당 주장에 귀속된 풀의 snippet_id 만 허용
        for v in verdicts:
            valid_ids = {
                e.snippet_id for e in pool if e.claim_id == v.claim_id
            }
            dropped = [s for s in v.evidence_chain if s not in valid_ids]
            if dropped:
                logger.warning("환각 evidence_chain 제거: %s (유효 id 아님)", dropped)
            kept = [s for s in v.evidence_chain if s in valid_ids]
            all_hallucinated = bool(dropped) and not kept
            v.evidence_chain = kept
            if v.label == VerdictLabel.MIXED:
                # 혼재는 종합 등급 전용. 주장 단위로 나오면 불충분 강등 후 신뢰도 상한 적용
                logger.warning("주장 단위 혼재 라벨 → 불충분 강등 (claim %d)", v.claim_id)
                v.label = VerdictLabel.INSUFFICIENT
                v.confidence = min(v.confidence, 0.5)
            if all_hallucinated and v.label in POLAR_LABELS:
                # 인용 전부가 환각 id 인 단정 판정은 불충분 강등
                logger.warning(
                    "전량 환각 인용의 단정 판정 → 불충분 강등 (claim %d)", v.claim_id
                )
                v.label = VerdictLabel.INSUFFICIENT
                v.confidence = min(v.confidence, 0.5)
            elif v.label in POLAR_LABELS and not v.evidence_chain:
                # 상식 예외 경로: 코퍼스로 검증 불가한 파라메트릭 판정이므로 신뢰도 상한
                v.confidence = min(v.confidence, 0.85)
    else:
        verdicts = []

    # 각 판정의 자가 반박으로 refutation_log 구성 및 추가검색 유도
    refutations = [
        RefutationEntry(
            loop=loop,
            claim_id=v.claim_id,
            challenge=v.self_refutation,
            survived=v.survives_refutation,
        )
        for v in verdicts
    ]
    for v in verdicts:
        ev_count = len(evidence_for_claim(pool, v.claim_id))
        if (not v.survives_refutation) and ev_count <= _THIN_EVIDENCE:
            v.needs_more_evidence = True

    new_loop = loop + 1
    avg_conf = (
        sum(v.confidence for v in verdicts) / len(verdicts)
        if verdicts
        else 0.0
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

    update = {
        "verdicts": verdicts,                 # 교체
        "refutation_log": refutations,        # add 리듀서
        "loop_count": new_loop,               # 카운터 교체
        "last_confidence": avg_conf,          # 다음 라운드 비교 기준
        "route_decision": decision,           # 라우터가 읽음
    }
    if decision == "retrieve_evidence":
        # 증거 재요청을 대화창에 가시화하는 판사 시스템 턴
        update["debate_transcript"] = [
            DebateTurn(
                claim_id=-1, loop=loop, turn=99, role="판사",
                text=(
                    "현재 증거로는 판단이 어렵습니다. 양측 리서처에게 "
                    f"추가 증거 수집을 지시합니다. (라운드 {new_loop + 1})"
                ),
            )
        ]
    return update


def route_after_judge(state: FactCheckState) -> str:
    """조건 분기 함수. judge 가 계산한 결정을 그대로 사용한다."""
    return state.get("route_decision", "synthesize")
