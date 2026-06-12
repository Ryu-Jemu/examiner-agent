"""synthesize 종합 등급 집계: 혼재 분리, 평균 경로, 판정 근거 표기."""

import importlib

from factchecker.models import (
    Claim,
    ClaimType,
    RebuttalCard,
    Verdict,
    VerdictLabel,
)

syn_mod = importlib.import_module("factchecker.nodes.synthesize")


def _claim(cid: int) -> Claim:
    return Claim(
        claim_id=cid, text=f"주장 {cid}",
        claim_type=ClaimType.FACT, checkable=True,
    )


def _verdict(cid, label: VerdictLabel, conf: float, chain=None) -> Verdict:
    return Verdict(
        claim_id=cid, label=label, confidence=conf,
        evidence_chain=chain or [],
    )


def _patch_rebuttal(monkeypatch):
    def fake_invoke(prompt, schema, *, default):
        return default  # LLM 없이 등급 일관 폴백 문구 사용

    monkeypatch.setattr(syn_mod, "structured_invoke", fake_invoke)


def _run(monkeypatch, verdicts):
    _patch_rebuttal(monkeypatch)
    state = {
        "claims": [_claim(v.claim_id) for v in verdicts],
        "verdicts": verdicts,
        "arguments": [],
        "evidence_pool": [],
        "technique_tags": [],
    }
    return syn_mod.synthesize(state)["final_report"]


def test_conflicting_verdicts_yield_mixed_not_insufficient(monkeypatch):
    # 회귀: 거짓(0.0)+사실(1.0) 평균 0.5 가 "불충분"으로 둔갑하던 집계 결함
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.FALSE, 0.95),
        _verdict(1, VerdictLabel.TRUE, 0.99),
    ])
    assert report.overall_grade == VerdictLabel.MIXED
    assert abs(report.overall_confidence - 0.97) < 1e-9


def test_low_confidence_polar_does_not_trigger_mixed(monkeypatch):
    # 저신뢰(0.5 미만) 단정 1건이 종합을 혼재로 뒤집지 못하고 평균 경로를 탄다
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.TRUE, 0.99),
        _verdict(1, VerdictLabel.MOSTLY_FALSE, 0.10),
    ])
    assert report.overall_grade == VerdictLabel.MOSTLY_TRUE


def test_all_insufficient_stays_insufficient(monkeypatch):
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.INSUFFICIENT, 0.3),
        _verdict(1, VerdictLabel.INSUFFICIENT, 0.4),
    ])
    assert report.overall_grade == VerdictLabel.INSUFFICIENT


def test_same_side_average_path_unchanged(monkeypatch):
    # 같은 방향(참 계열)끼리는 기존 평균 집계 유지
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.TRUE, 0.9),
        _verdict(1, VerdictLabel.MOSTLY_TRUE, 0.8),
    ])
    assert report.overall_grade == VerdictLabel.TRUE


def test_breakdown_basis_labels(monkeypatch):
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.TRUE, 0.9, chain=["ev_a", "ev_b"]),
        _verdict(1, VerdictLabel.FALSE, 0.8),
        _verdict(2, VerdictLabel.INSUFFICIENT, 0.3),
    ])
    basis = {b.claim_id: b.basis for b in report.claim_breakdown}
    assert basis[0] == "코퍼스 증거 2건"
    assert basis[1] == "일반 상식 기반(코퍼스 인용 없음)"
    assert basis[2] == "관련 증거 부족"


def test_mixed_fallback_rebuttal_is_consistent(monkeypatch):
    report = _run(monkeypatch, [
        _verdict(0, VerdictLabel.FALSE, 0.9),
        _verdict(1, VerdictLabel.TRUE, 0.9),
    ])
    assert "섞여" in report.rebuttal_card


def test_rebuttal_default_type(monkeypatch):
    # 폴백 카드가 RebuttalCard 스키마로 만들어지는지(구조 보존)
    _patch_rebuttal(monkeypatch)
    fallback = syn_mod._fallback_rebuttal(VerdictLabel.MIXED)
    assert RebuttalCard(rebuttal_card=fallback).rebuttal_card
