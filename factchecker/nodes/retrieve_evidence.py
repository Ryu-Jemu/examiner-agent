"""Node 2: retrieve_evidence (RAG) — 주장별 증거를 회수해 공유 풀에 누적.

LLM 을 쓰지 않아 결정론적이다. 이미 풀에 있는 스니펫은 dedup 하고 신규만 반환하여
`add` 리듀서가 풀을 누적하게 한다. `prev_pool_size` 를 갱신해 루프 종료 판정을 돕는다.
"""

from __future__ import annotations

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
    except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            logger.warning("주장 %d 증거 회수 실패: %s", claim.claim_id, exc)
            items = []
        new_items.extend(items)

    logger.info("신규 증거 %d개 회수(이전 풀 %d개)", len(new_items), prev_size)
    # evidence_pool 은 add 리듀서 → 신규만 반환. prev_pool_size 는 교체(단일 쓰기).
    return {"evidence_pool": new_items, "prev_pool_size": prev_size}
