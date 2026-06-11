"""judge — 재검색 루프 시 '판사' 시스템 턴이 트랜스크립트에 추가되는지."""

from factchecker.models import (
    Claim,
    ClaimType,
    EvidenceItem,
    SourceType,
    Verdict,
    VerdictLabel,
    VerdictList,
)
import importlib

# nodes/__init__ 이 동명 함수를 재수출하므로 모듈 객체를 직접 가져온다
judge_mod = importlib.import_module("factchecker.nodes.judge")


def _state(pool_size: int = 1):
    claim = Claim(
        claim_id=0, text="테스트 주장",
        claim_type=ClaimType.FACT, checkable=True,
    )
    pool = [
        EvidenceItem(
            claim_id=0, snippet_id=f"ev_{i}", snippet="내용",
            source="테스트", source_type=SourceType.NEWS, credibility=0.7,
        )
        for i in range(pool_size)
    ]
    return {
        "claims": [claim], "evidence_pool": pool, "arguments": [],
        "loop_count": 0, "last_confidence": 0.0, "prev_pool_size": -1,
    }


def _patch_verdict(monkeypatch, needs_more: bool):
    def fake_invoke(prompt, schema, *, default):
        return VerdictList(verdicts=[
            Verdict(
                claim_id=0, label=VerdictLabel.INSUFFICIENT, confidence=0.3,
                evidence_chain=[], needs_more_evidence=needs_more,
            )
        ])

    monkeypatch.setattr(judge_mod, "structured_invoke", fake_invoke)


def test_loop_emits_judge_system_turn(monkeypatch):
    _patch_verdict(monkeypatch, needs_more=True)
    update = judge_mod.judge(_state())
    assert update["route_decision"] == "retrieve_evidence"
    turns = update["debate_transcript"]
    assert len(turns) == 1 and turns[0].role == "판사"
    assert "추가 증거" in turns[0].text


def test_no_system_turn_when_satisfied(monkeypatch):
    _patch_verdict(monkeypatch, needs_more=False)
    update = judge_mod.judge(_state())
    assert update["route_decision"] == "synthesize"
    assert "debate_transcript" not in update
