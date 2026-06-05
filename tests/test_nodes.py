"""노드 단위 테스트 (structured_invoke / RAG 를 모킹해 네트워크·키 불필요)."""

from __future__ import annotations

import importlib

from factchecker.models import (
    ArgumentPair,
    Claim,
    ClaimList,
    ClaimType,
    EvidenceItem,
    RebuttalCard,
    RefutationEntry,
    RefutationList,
    SideArgument,
    SourceType,
    TechniqueTag,
    TechniqueTagList,
    TechniqueTagName,
    Verdict,
    VerdictLabel,
    VerdictList,
)


def test_extract_empty_input_returns_no_claims():
    from factchecker.nodes.extract_claims import extract_claims

    out = extract_claims({"input_text": "   "})
    assert out["claims"] == []
    assert out["loop_count"] == 0
    assert out["prev_pool_size"] == 0


def test_extract_renumbers_claim_ids(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.extract_claims")

    def fake_si(prompt, schema, *, default, temperature=None):
        return ClaimList(
            claims=[
                Claim(claim_id=5, text="a", claim_type=ClaimType.FACT, checkable=True),
                Claim(claim_id=9, text="b", claim_type=ClaimType.OPINION, checkable=False),
            ]
        )

    monkeypatch.setattr(mod, "structured_invoke", fake_si)
    out = mod.extract_claims({"input_text": "테스트"})
    assert [c.claim_id for c in out["claims"]] == [0, 1]


def test_debate_drops_hallucinated_citations(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.adversarial_debate")

    def fake_si(prompt, schema, *, default, temperature=None):
        return SideArgument(summary="x", cited_snippet_ids=["ev1", "MADE_UP"])

    monkeypatch.setattr(mod, "structured_invoke", fake_si)
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    pool = [EvidenceItem(claim_id=0, snippet_id="ev1", snippet="근거", source="s")]
    out = mod.adversarial_debate(
        {"claims": claims, "evidence_pool": pool, "loop_count": 0}
    )
    ap = out["arguments"][0]
    assert ap.prosecution.cited_snippet_ids == ["ev1"]
    assert ap.defense.cited_snippet_ids == ["ev1"]


def test_judge_falls_back_to_insufficient(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.judge")

    def fake_si(prompt, schema, *, default, temperature=None):
        return default  # LLM 실패 시뮬레이션

    monkeypatch.setattr(mod, "structured_invoke", fake_si)
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    out = mod.judge(
        {"claims": claims, "evidence_pool": [], "arguments": [], "loop_count": 0}
    )
    assert out["verdicts"][0].label == VerdictLabel.INSUFFICIENT
    assert out["loop_count"] == 1
    assert out["route_decision"] in ("retrieve_evidence", "synthesize")


def test_synthesize_deterministic_grade(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.synthesize")

    monkeypatch.setattr(
        mod,
        "structured_invoke",
        lambda *a, **k: RebuttalCard(rebuttal_card="msg"),
    )
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    verdicts = [Verdict(claim_id=0, label=VerdictLabel.FALSE, confidence=0.9)]
    args = [
        ArgumentPair(
            claim_id=0,
            loop=0,
            prosecution=SideArgument(summary="p", cited_snippet_ids=["ev1"]),
            defense=SideArgument(summary="d", cited_snippet_ids=[]),
        )
    ]
    pool = [
        EvidenceItem(
            claim_id=0, snippet_id="ev1", snippet="근거",
            source="출처A", source_type=SourceType.NEWS,
        )
    ]
    out = mod.synthesize(
        {"claims": claims, "verdicts": verdicts, "arguments": args, "evidence_pool": pool}
    )
    rep = out["final_report"]
    assert rep.overall_grade == VerdictLabel.FALSE
    assert rep.claim_breakdown[0].refuting_sources == ["출처A (뉴스)"]
    assert rep.rebuttal_card == "msg"


def test_synthesize_empty_is_insufficient(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.synthesize")

    monkeypatch.setattr(
        mod, "structured_invoke", lambda *a, **k: RebuttalCard(rebuttal_card="m")
    )
    out = mod.synthesize({"claims": [], "verdicts": [], "arguments": [], "evidence_pool": []})
    assert out["final_report"].overall_grade == VerdictLabel.INSUFFICIENT


def test_tag_techniques_corrects_library_id(monkeypatch):
    mod = importlib.import_module("factchecker.nodes.tag_techniques")

    monkeypatch.setattr(
        mod,
        "retrieve_techniques",
        lambda text: [
            {"library_entry_id": "tech_emotion", "tag": "감정 자극", "content": "..."}
        ],
    )

    def fake_si(prompt, schema, *, default, temperature=None):
        return TechniqueTagList(
            technique_tags=[
                TechniqueTag(
                    tag=TechniqueTagName.EMOTION,
                    evidence_sentence="충격!",
                    library_entry_id="WRONG",
                    confidence=0.8,
                )
            ]
        )

    monkeypatch.setattr(mod, "structured_invoke", fake_si)
    out = mod.tag_techniques({"input_text": "충격! 큰일났습니다"})
    assert out["technique_tags"][0].library_entry_id == "tech_emotion"


def _schema_aware_judge_si(verdict, refute):
    def fake_si(prompt, schema, *, default, temperature=None):
        if schema is VerdictList:
            return VerdictList(verdicts=[verdict])
        if schema is RefutationList:
            return RefutationList(refutations=[refute])
        return default
    return fake_si


def test_judge_escalates_needs_more_on_failed_refutation(monkeypatch):
    """자가 반박 미생존 + 증거 빈약 → needs_more_evidence=True → 루프."""
    mod = importlib.import_module("factchecker.nodes.judge")
    verdict = Verdict(
        claim_id=0, label=VerdictLabel.INSUFFICIENT, confidence=0.3,
        evidence_chain=["ev1"], needs_more_evidence=False,
    )
    refute = RefutationEntry(loop=0, claim_id=0, challenge="반론", survived=False)
    monkeypatch.setattr(mod, "structured_invoke", _schema_aware_judge_si(verdict, refute))
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    pool = [EvidenceItem(claim_id=0, snippet_id="ev1", snippet="x", source="s")]  # 1건=빈약
    out = mod.judge(
        {"claims": claims, "evidence_pool": pool, "arguments": [],
         "loop_count": 0, "prev_pool_size": 0}
    )
    assert out["verdicts"][0].needs_more_evidence is True
    assert out["route_decision"] == "retrieve_evidence"


def test_judge_no_escalation_when_evidence_sufficient(monkeypatch):
    """증거가 충분(>1건)하면 미생존이어도 추가검색을 유도하지 않는다."""
    mod = importlib.import_module("factchecker.nodes.judge")
    verdict = Verdict(
        claim_id=0, label=VerdictLabel.MOSTLY_TRUE, confidence=0.8,
        evidence_chain=["ev1", "ev2"], needs_more_evidence=False,
    )
    refute = RefutationEntry(loop=0, claim_id=0, challenge="반론", survived=False)
    monkeypatch.setattr(mod, "structured_invoke", _schema_aware_judge_si(verdict, refute))
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    pool = [
        EvidenceItem(claim_id=0, snippet_id="ev1", snippet="x", source="s"),
        EvidenceItem(claim_id=0, snippet_id="ev2", snippet="y", source="s"),
    ]
    out = mod.judge(
        {"claims": claims, "evidence_pool": pool, "arguments": [],
         "loop_count": 0, "prev_pool_size": 0}
    )
    assert out["verdicts"][0].needs_more_evidence is False
    assert out["route_decision"] == "synthesize"


def test_judge_filters_hallucinated_evidence_chain(monkeypatch):
    """판사가 풀에 없는 snippet_id 를 evidence_chain 에 넣으면 제거된다(환각 가드)."""
    mod = importlib.import_module("factchecker.nodes.judge")
    verdict = Verdict(
        claim_id=0, label=VerdictLabel.FALSE, confidence=0.9,
        evidence_chain=["ev1", "FAKE"], needs_more_evidence=False,
    )
    refute = RefutationEntry(loop=0, claim_id=0, challenge="r", survived=True)
    monkeypatch.setattr(mod, "structured_invoke", _schema_aware_judge_si(verdict, refute))
    claims = [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)]
    pool = [EvidenceItem(claim_id=0, snippet_id="ev1", snippet="x", source="s")]
    out = mod.judge(
        {"claims": claims, "evidence_pool": pool, "arguments": [],
         "loop_count": 0, "prev_pool_size": 0}
    )
    assert out["verdicts"][0].evidence_chain == ["ev1"]  # FAKE 제거됨


def test_retrieve_evidence_node_skips_opinions_and_sets_prev_size(monkeypatch):
    """검증대상만 회수, prev_pool_size=직전 풀 크기, 신규만 반환."""
    import factchecker.rag.vectorstore as vs
    mod = importlib.import_module("factchecker.nodes.retrieve_evidence")
    monkeypatch.setattr(vs, "get_or_build_evidence", lambda: object())
    seen = {}

    def fake_retrieve(text, cid, *, k, store, existing_ids):
        seen["existing"] = set(existing_ids)
        return [EvidenceItem(claim_id=cid, snippet_id=f"new{cid}", snippet="n", source="s")]

    monkeypatch.setattr(mod, "retrieve_for_claim", fake_retrieve)
    claims = [
        Claim(claim_id=0, text="a", claim_type=ClaimType.FACT, checkable=True),
        Claim(claim_id=1, text="b", claim_type=ClaimType.OPINION, checkable=False),
    ]
    pool = [EvidenceItem(claim_id=0, snippet_id="ev1", snippet="x", source="s")]
    out = mod.retrieve_evidence({"claims": claims, "evidence_pool": pool})
    assert out["prev_pool_size"] == 1
    assert {e.snippet_id for e in out["evidence_pool"]} == {"new0"}  # 의견(claim 1) 제외
    assert (0, "ev1") in seen["existing"]  # 복합 키로 전달
