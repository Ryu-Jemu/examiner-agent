"""증거 회수.

로컬 Chroma `evidence` 컬렉션에서 주장 텍스트와 유사한 스니펫을 회수하고,
설정에 따라 선택적으로 라이브 웹 검색 결과를 추가한다. 회수 결과는 EvidenceItem
으로 변환되며, 신뢰도는 source_type 기반으로 결정론적으로 부여된다.
"""

from __future__ import annotations

import logging

from ..config import get_settings
from ..models import SOURCE_CREDIBILITY, EvidenceItem, SourceType

logger = logging.getLogger("factchecker.rag.evidence")


def _source_type_from_str(value: str) -> SourceType:
    for st in SourceType:
        if st.value == value:
            return st
    return SourceType.UNKNOWN


def _doc_to_evidence(doc, claim_id: int) -> EvidenceItem:
    meta = doc.metadata or {}
    st = _source_type_from_str(meta.get("source_type", "출처불명"))
    return EvidenceItem(
        claim_id=claim_id,
        snippet_id=str(meta.get("snippet_id", "")),
        snippet=doc.page_content,
        source=meta.get("source", ""),
        source_type=st,
        credibility=SOURCE_CREDIBILITY[st],
        url=(meta.get("url") or None),
    )


def retrieve_for_claim(
    claim_text: str,
    claim_id: int,
    *,
    k: int | None = None,
    store=None,
    existing_ids: set | None = None,
) -> list[EvidenceItem]:
    """주장 하나에 대한 증거 스니펫 목록을 회수한다.

    `existing_ids` 는 **(claim_id, snippet_id) 복합 키** 집합이다. 같은 스니펫이라도
    서로 다른 주장에는 각각 귀속될 수 있어야 하므로(공유 증거), 동일 *주장 내* 중복만
    제외한다.
    """
    settings = get_settings()
    k = settings.retrieve_k if k is None else k
    if existing_ids is None:  # 빈 set 도 그대로 사용·변이(호출 간 공유)되도록
        existing_ids = set()

    if store is None:
        from .vectorstore import get_or_build_evidence

        store = get_or_build_evidence()

    items: list[EvidenceItem] = []

    # 1) 로컬 코퍼스 (항상 사용 — 결정론적)
    try:
        docs = store.similarity_search(claim_text, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.warning("로컬 유사도 검색 실패: %s", exc)
        docs = []

    for doc in docs:
        item = _doc_to_evidence(doc, claim_id)
        key = (claim_id, item.snippet_id or item.snippet[:40])
        if key in existing_ids:
            continue
        existing_ids.add(key)
        items.append(item)

    # 2) 선택적 라이브 웹 검색
    if settings.search_backend != "local":
        from .web_search import search_web

        for hit in search_web(claim_text, k=k):
            sid = hit["snippet_id"]
            key = (claim_id, sid)
            if key in existing_ids:
                continue
            existing_ids.add(key)
            items.append(
                EvidenceItem(
                    claim_id=claim_id,
                    snippet_id=sid,
                    snippet=hit["snippet"],
                    source=hit.get("source", "웹 검색"),
                    source_type=SourceType.UNKNOWN,
                    credibility=SOURCE_CREDIBILITY[SourceType.UNKNOWN],
                    url=hit.get("url"),
                )
            )

    return items
