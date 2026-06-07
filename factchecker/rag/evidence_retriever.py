"""로컬 Chroma evidence 컬렉션에서 주장과 유사한 증거 스니펫을 회수한다."""

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
    """주장 하나의 증거 스니펫을 회수한다. existing_ids 는 (claim_id, snippet_id) 복합
    키 집합으로 동일 주장 내 중복만 제외한다(같은 스니펫이 다른 주장엔 각각 귀속)."""
    settings = get_settings()
    k = settings.retrieve_k if k is None else k
    if existing_ids is None:  # 빈 set 도 그대로 사용·변이(호출 간 공유)되도록
        existing_ids = set()

    if store is None:
        from .vectorstore import get_or_build_evidence

        store = get_or_build_evidence()

    items: list[EvidenceItem] = []

    # 관련성(코사인) 임계값 이상만 채택해 코퍼스 밖 주장에 엉뚱한 근거가 섞이지 않게 한다.
    # 임계값<=0 이면 점수 없이 회수한다(결정론적 테스트용).
    min_rel = settings.retrieve_min_relevance
    scored: list[tuple] = []
    try:
        if min_rel > 0:
            scored = list(store.similarity_search_with_relevance_scores(claim_text, k=k))
        else:
            scored = [(doc, None) for doc in store.similarity_search(claim_text, k=k)]
    except Exception as exc:
        logger.warning("로컬 유사도 검색 실패: %s", exc)
        scored = []

    for doc, score in scored:
        if score is not None and score < min_rel:
            continue  # 관련성 낮음 → 무관 스니펫 제외
        item = _doc_to_evidence(doc, claim_id)
        key = (claim_id, item.snippet_id or item.snippet[:40])
        if key in existing_ids:
            continue
        existing_ids.add(key)
        items.append(item)

    return items
