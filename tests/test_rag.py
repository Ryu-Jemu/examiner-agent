"""RAG 테스트: 인덱스 빌드/멱등성, 증거 회수 (가짜 임베딩 사용)."""

from __future__ import annotations

import json


def test_evidence_index_builds_and_counts_match(fake_embeddings):
    from factchecker.config import get_settings
    from factchecker.rag.vectorstore import get_or_build_evidence

    store = get_or_build_evidence(embeddings=fake_embeddings)
    corpus = json.loads(get_settings().evidence_corpus_path.read_text(encoding="utf-8"))
    assert store._collection.count() == len(corpus)  # noqa: SLF001


def test_ingest_is_idempotent_via_manifest(fake_embeddings):
    from factchecker.config import get_settings
    from factchecker.rag.vectorstore import EVIDENCE_COLLECTION, get_or_build_evidence

    get_or_build_evidence(embeddings=fake_embeddings)
    manifest_path = get_settings().chroma_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert EVIDENCE_COLLECTION in manifest

    # 두 번째 호출은 해시 일치 → 동일 개수, 오류 없음
    store2 = get_or_build_evidence(embeddings=fake_embeddings)
    corpus = json.loads(get_settings().evidence_corpus_path.read_text(encoding="utf-8"))
    assert store2._collection.count() == len(corpus)  # noqa: SLF001


def test_retrieve_returns_evidence_items_with_credibility(fake_embeddings):
    from factchecker.models import SOURCE_CREDIBILITY, EvidenceItem
    from factchecker.rag.evidence_retriever import retrieve_for_claim
    from factchecker.rag.vectorstore import get_or_build_evidence

    store = get_or_build_evidence(embeddings=fake_embeddings)
    items = retrieve_for_claim("백신 자석", claim_id=3, k=4, store=store)
    assert items, "증거가 회수되어야 함"
    for it in items:
        assert isinstance(it, EvidenceItem)
        assert it.claim_id == 3
        assert it.snippet_id
        assert it.credibility == SOURCE_CREDIBILITY[it.source_type]


def test_retrieve_dedup_excludes_existing(fake_embeddings):
    from factchecker.rag.evidence_retriever import retrieve_for_claim
    from factchecker.rag.vectorstore import get_or_build_evidence

    store = get_or_build_evidence(embeddings=fake_embeddings)
    first = retrieve_for_claim("백신 자석", claim_id=0, k=4, store=store)
    # 복합 키 (claim_id, snippet_id) 로 동일 주장 중복을 제외
    existing = {(0, e.snippet_id) for e in first}
    again = retrieve_for_claim(
        "백신 자석", claim_id=0, k=4, store=store, existing_ids=set(existing)
    )
    seen_ids = {e.snippet_id for e in first}
    assert all(e.snippet_id not in seen_ids for e in again)


def test_shared_snippet_attaches_to_multiple_claims():
    """공유 스니펫이 여러 주장에 각각 귀속되고, 동일 주장 반복만 dedup 된다."""
    from langchain_core.documents import Document

    from factchecker.rag.evidence_retriever import retrieve_for_claim

    class FakeStore:
        def similarity_search(self, query, k):
            return [
                Document(
                    page_content="공유 근거",
                    metadata={"snippet_id": "S1", "source": "s",
                              "source_type": "학술", "url": ""},
                )
            ]

    store = FakeStore()
    shared: set = set()
    a = retrieve_for_claim("주장0", 0, k=2, store=store, existing_ids=shared)
    b = retrieve_for_claim("주장1", 1, k=2, store=store, existing_ids=shared)
    # 같은 스니펫 S1 이 claim 0 과 claim 1 모두에 귀속되어야 함
    assert any(e.claim_id == 0 and e.snippet_id == "S1" for e in a)
    assert any(e.claim_id == 1 and e.snippet_id == "S1" for e in b)
    # 동일 주장(claim 0) 재회수는 dedup 되어 빈 목록
    again = retrieve_for_claim("주장0", 0, k=2, store=store, existing_ids=shared)
    assert again == []


def test_technique_index_builds(fake_embeddings):
    from factchecker.rag.technique_retriever import retrieve_techniques
    from factchecker.rag.vectorstore import get_or_build_techniques

    store = get_or_build_techniques(embeddings=fake_embeddings)
    assert store._collection.count() == 4  # noqa: SLF001
    entries = retrieve_techniques("충격! 큰일났습니다", store=store, k=4)
    assert len(entries) == 4
    assert all("library_entry_id" in e and "tag" in e for e in entries)
