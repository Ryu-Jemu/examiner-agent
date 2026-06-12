"""주장별 증거를 회수해 공유 풀에 누적하는 노드(결정론적, LLM 미사용)."""

import logging

from ..config import get_settings
from ..models import Claim, DebateTurn, EvidenceItem
from ..rag.evidence_retriever import retrieve_for_claim
from ..state import FactCheckState

logger = logging.getLogger("factchecker.nodes.retrieve")


def retrieve_evidence(state: FactCheckState) -> dict:
    settings = get_settings()
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    loop = state.get("loop_count", 0)

    # 동일 주장 내 중복만 방지, 같은 스니펫의 타 주장 귀속은 허용
    existing_ids: set = {
        (e.claim_id, e.snippet_id or e.snippet[:40]) for e in pool
    }
    prev_size = len(pool)

    # 벡터스토어 로드 실패는 코퍼스 증거 없음과 구분되는 인프라 장애로 플래그
    failed = False
    store = None
    try:
        from ..rag.vectorstore import get_or_build_evidence

        store = get_or_build_evidence()
    except Exception as exc:
        logger.warning("증거 벡터스토어 로드 실패: %s", exc)
        failed = True
        store = None

    new_items: list[EvidenceItem] = []
    for claim in claims:
        if not claim.checkable:
            continue
        try:
            items = retrieve_for_claim(
                claim.text,
                claim.claim_id,
                k=settings.retrieve_k,
                store=store,
                existing_ids=existing_ids,
            )
        except Exception as exc:
            logger.warning("주장 %d 증거 회수 실패: %s", claim.claim_id, exc)
            failed = True
            items = []
        new_items.extend(items)

    # evidence_pool 은 add 리듀서이므로 신규만 반환
    update = {
        "evidence_pool": new_items,
        "prev_pool_size": prev_size,
        "retrieval_failed": failed,
    }
    if loop > 0 and not new_items:
        # 재검색에 신규 증거가 없을 때 대화창 마감 안내
        update["debate_transcript"] = [
            DebateTurn(
                claim_id=-1, loop=loop, turn=99, role="판사",
                text="재검색 결과 신규 증거가 없어 직전 판정으로 종합합니다.",
            )
        ]
    return update


def route_after_retrieve(state: FactCheckState) -> str:
    """재검색에서 신규 증거가 0건이면 토론·판사를 건너뛰고 합성으로 보낸다."""
    loop = state.get("loop_count", 0)
    pool = state.get("evidence_pool", []) or []
    prev = state.get("prev_pool_size", -1)
    if loop > 0 and len(pool) == prev:
        return "synthesize"
    return "adversarial_debate"
