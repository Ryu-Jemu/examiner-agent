"""evidence_retriever 양측 스탠스 쿼리, both 승격, 라운드 간 중복 제거."""

from factchecker.models import (
    STANCE_BOTH,
    STANCE_DEFENSE,
    STANCE_PROSECUTION,
)
from factchecker.rag.evidence_retriever import retrieve_for_claim


class _Doc:
    def __init__(self, sid: str):
        self.page_content = f"{sid} 내용"
        self.metadata = {
            "snippet_id": sid, "source": "테스트", "source_type": "학술",
        }


class _Store:
    """쿼리 문구(반박/지지)에 따라 다른 스니펫을 돌려주는 스텁."""

    def __init__(self, pro_ids, def_ids):
        self.pro_ids, self.def_ids = pro_ids, def_ids

    def similarity_search_with_relevance_scores(self, query, k):
        ids = self.pro_ids if "거짓" in query else self.def_ids
        return [(_Doc(s), 0.9) for s in ids[:k]]


def test_stance_tagging_and_both_upgrade():
    store = _Store(pro_ids=["ev_a", "ev_shared"], def_ids=["ev_shared", "ev_b"])
    items = retrieve_for_claim("주장", 0, k=3, store=store, existing_ids=set())
    by_id = {i.snippet_id: i for i in items}
    assert by_id["ev_a"].stance == STANCE_PROSECUTION
    assert by_id["ev_b"].stance == STANCE_DEFENSE
    assert by_id["ev_shared"].stance == STANCE_BOTH


def test_existing_ids_dedup_across_rounds():
    store = _Store(pro_ids=["ev_a"], def_ids=["ev_b"])
    seen: set = set()
    first = retrieve_for_claim("주장", 0, k=3, store=store, existing_ids=seen)
    assert {i.snippet_id for i in first} == {"ev_a", "ev_b"}
    # 같은 라운드 키 집합을 공유하면 재검색에 신규가 없어 종료조건(3) 입력이 된다
    second = retrieve_for_claim("주장", 0, k=3, store=store, existing_ids=seen)
    assert second == []


class _ScoredStore:
    """주제 일치(고점수)·무관(저점수) 스니펫을 함께 돌려주는 스텁.

    실측 분포(gemini-embedding-001·cosine): 주제 일치 0.717 이상,
    무관 0.676 이하. 기본 임계값 0.70 이 그 사이를 가른다.
    """

    def similarity_search_with_relevance_scores(self, query, k):
        on_topic = _Doc("ev_on_topic")
        on_topic.metadata["date"] = "2021"
        return [(on_topic, 0.74), (_Doc("ev_off_topic"), 0.62)]


def test_min_relevance_filters_off_topic_snippets():
    items = retrieve_for_claim(
        "주장", 0, k=3, store=_ScoredStore(), existing_ids=set()
    )
    # 무관 스니펫(0.62)은 기본 임계값 0.70 에 걸러지고 주제 일치만 남는다
    assert {i.snippet_id for i in items} == {"ev_on_topic"}


def test_relevance_and_date_propagated_to_evidence():
    items = retrieve_for_claim(
        "주장", 0, k=3, store=_ScoredStore(), existing_ids=set()
    )
    item = items[0]
    assert item.relevance == 0.74
    assert item.date == "2021"
