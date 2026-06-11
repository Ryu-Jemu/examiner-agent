"""adversarial_debate — 3턴 토론, 환각 인용 필터, 트랜스크립트 생성."""

from factchecker.models import (
    STANCE_BOTH,
    STANCE_DEFENSE,
    STANCE_PROSECUTION,
    Claim,
    ClaimType,
    EvidenceItem,
    SideArgument,
    SourceType,
)
import importlib

# nodes/__init__ 이 동명 함수를 재수출하므로 모듈 객체를 직접 가져온다
deb = importlib.import_module("factchecker.nodes.adversarial_debate")
_filter_citations = deb._filter_citations


def _claim(cid: int = 0) -> Claim:
    return Claim(
        claim_id=cid, text="백신을 맞으면 자석이 붙는다",
        claim_type=ClaimType.FACT, checkable=True,
    )


def _ev(sid: str, stance: str | None, cid: int = 0) -> EvidenceItem:
    return EvidenceItem(
        claim_id=cid, snippet_id=sid, snippet=f"{sid} 내용",
        source="테스트 출처", source_type=SourceType.ACADEMIC,
        credibility=0.85, stance=stance,
    )


def test_filter_citations_drops_invented_ids():
    side = SideArgument(summary="요약", cited_snippet_ids=["ev_1", "ev_fake"])
    out = _filter_citations(side, {"ev_1"})
    assert out.cited_snippet_ids == ["ev_1"]
    assert out.summary == "요약"


def test_three_turns_and_transcript(monkeypatch):
    prompts_seen: list[str] = []

    def fake_invoke(prompt, schema, *, default):
        prompts_seen.append(prompt)
        n = len(prompts_seen)
        # 검사·재반박은 검사 풀(ev_pro)에 없는 id 를 섞어 필터 검증
        return SideArgument(
            summary=f"발언{n}", cited_snippet_ids=["ev_pro", "ev_def"]
        )

    monkeypatch.setattr(deb, "structured_invoke", fake_invoke)
    pool = [
        _ev("ev_pro", STANCE_PROSECUTION),
        _ev("ev_def", STANCE_DEFENSE),
        _ev("ev_both", STANCE_BOTH),
    ]
    update = deb.adversarial_debate(
        {"claims": [_claim()], "evidence_pool": pool, "loop_count": 0}
    )
    # LLM 3회 호출(검사 → 변호 → 검사 재반박)
    assert len(prompts_seen) == 3
    # 변호 프롬프트에는 검사 논거가, 재반박 프롬프트에는 변호 논거가 들어간다
    assert "발언1" in prompts_seen[1]
    assert "발언2" in prompts_seen[2]

    [pair] = update["arguments"]
    assert pair.prosecution_rebuttal is not None
    # 측별 환각 인용 필터: 검사는 자기 풀(ev_pro, ev_both)만 인용 가능
    assert pair.prosecution.cited_snippet_ids == ["ev_pro"]
    assert pair.defense.cited_snippet_ids == ["ev_def"]
    assert pair.prosecution_rebuttal.cited_snippet_ids == ["ev_pro"]

    turns = update["debate_transcript"]
    assert [t.role for t in turns] == ["검사", "변호", "검사"]
    assert [t.turn for t in turns] == [0, 1, 2]
    assert all(t.loop == 0 for t in turns)


def test_no_evidence_skips_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("증거가 없으면 LLM 을 호출하면 안 된다")

    monkeypatch.setattr(deb, "structured_invoke", boom)
    update = deb.adversarial_debate(
        {"claims": [_claim()], "evidence_pool": [], "loop_count": 0}
    )
    [pair] = update["arguments"]
    assert pair.prosecution.summary == "(인용할 증거 없음)"
    assert [t.role for t in update["debate_transcript"]] == ["검사", "변호"]
