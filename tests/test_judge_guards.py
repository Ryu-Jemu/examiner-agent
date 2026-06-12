"""judge — 판정 코드 가드: 전량 환각 인용 강등, 상식 판정 신뢰도 상한, 혼재 차단."""

import importlib

from factchecker.models import (
    Claim,
    ClaimType,
    EvidenceItem,
    SourceType,
    Verdict,
    VerdictLabel,
    VerdictList,
)

judge_mod = importlib.import_module("factchecker.nodes.judge")


def _state():
    claims = [
        Claim(claim_id=i, text=f"주장 {i}",
              claim_type=ClaimType.FACT, checkable=True)
        for i in range(3)
    ]
    pool = [
        EvidenceItem(
            claim_id=0, snippet_id="ev_real", snippet="내용",
            source="테스트", source_type=SourceType.NEWS, credibility=0.7,
        )
    ]
    return {
        "claims": claims, "evidence_pool": pool, "arguments": [],
        "loop_count": 0, "last_confidence": 0.0, "prev_pool_size": -1,
    }


def _patch(monkeypatch, verdicts):
    def fake_invoke(prompt, schema, *, default):
        return VerdictList(verdicts=verdicts)

    monkeypatch.setattr(judge_mod, "structured_invoke", fake_invoke)


def test_fully_hallucinated_citation_demotes_to_insufficient(monkeypatch):
    # 인용 전부가 풀에 없는 id 인 단정 판정 → 근거 없는 단정으로 보고 강등
    _patch(monkeypatch, [
        Verdict(claim_id=0, label=VerdictLabel.FALSE, confidence=0.95,
                evidence_chain=["ev_fake_1", "ev_fake_2"]),
    ])
    update = judge_mod.judge(_state())
    v = update["verdicts"][0]
    assert v.label == VerdictLabel.INSUFFICIENT
    assert v.confidence <= 0.5
    assert v.evidence_chain == []


def test_evidence_free_polar_verdict_confidence_capped(monkeypatch):
    # 상식 예외 경로(인용 없음)의 단정 판정은 코퍼스로 검증 불가 → 상한 0.85
    _patch(monkeypatch, [
        Verdict(claim_id=0, label=VerdictLabel.TRUE, confidence=0.99,
                evidence_chain=[]),
    ])
    update = judge_mod.judge(_state())
    v = update["verdicts"][0]
    assert v.label == VerdictLabel.TRUE  # 라벨은 유지
    assert v.confidence == 0.85


def test_cited_polar_verdict_keeps_confidence(monkeypatch):
    # 실제 풀 인용이 있는 판정은 신뢰도를 건드리지 않는다
    _patch(monkeypatch, [
        Verdict(claim_id=0, label=VerdictLabel.FALSE, confidence=0.95,
                evidence_chain=["ev_real"]),
    ])
    update = judge_mod.judge(_state())
    v = update["verdicts"][0]
    assert v.label == VerdictLabel.FALSE
    assert v.confidence == 0.95
    assert v.evidence_chain == ["ev_real"]


def test_claim_level_mixed_label_demoted(monkeypatch):
    # "혼재"는 종합 등급 전용 — 주장 단위로 나오면 불충분으로 강등
    _patch(monkeypatch, [
        Verdict(claim_id=0, label=VerdictLabel.MIXED, confidence=0.7,
                evidence_chain=[]),
    ])
    update = judge_mod.judge(_state())
    assert update["verdicts"][0].label == VerdictLabel.INSUFFICIENT
