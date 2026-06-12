"""주장별 증거를 회수해 공유 풀에 누적하는 노드(결정론적, LLM 미사용)."""

import logging

from ..config import get_settings
from ..models import Claim, EvidenceItem
from ..rag.evidence_retriever import retrieve_for_claim
from ..state import FactCheckState

logger = logging.getLogger("factchecker.nodes.retrieve")


def retrieve_evidence(state: FactCheckState) -> dict:
    settings = get_settings()
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []

    # 현재 풀의 (주장, 스니펫) 복합 식별자 — 동일 주장 내 중복만 방지하고
    # 같은 스니펫이 다른 주장에는 각각 귀속될 수 있게 한다(공유 증거).
    existing_ids: set = {
        (e.claim_id, e.snippet_id or e.snippet[:40]) for e in pool
    }
    prev_size = len(pool)

    # 증거 벡터스토어를 한 번만 로드해 재사용
    store = None
    try:
        from ..rag.vectorstore import get_or_build_evidence

        store = get_or_build_evidence()
    except Exception as exc:
        logger.warning("증거 벡터스토어 로드 실패: %s", exc)
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
            items = []
        new_items.extend(items)

    # evidence_pool 은 add 리듀서 → 신규만 반환. prev_pool_size 는 단일 쓰기.
    return {"evidence_pool": new_items, "prev_pool_size": prev_size}


def route_after_retrieve(state: FactCheckState) -> str:
    """재검색(2라운드 이후)에서 신규 증거가 0건이면 토론·판사를 건너뛴다.

    스탠스 쿼리는 주장 텍스트만의 함수라 재검색 결과가 결정론적으로 동일하고,
    전부 중복 제거되어 풀이 정체된다. 이때 토론·판사를 다시 돌려도 입력이
    같아 정보가 늘지 않으므로(직전 라운드 판정 유지) 바로 합성으로 보낸다
    — 주장당 토론 3회 + 판사 1회의 LLM 호출 절약.
    """
    loop = state.get("loop_count", 0)
    pool = state.get("evidence_pool", []) or []
    prev = state.get("prev_pool_size", -1)
    if loop > 0 and len(pool) == prev:
        return "synthesize"
    return "adversarial_debate"
