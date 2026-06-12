"""adversarial_debate: 3턴 토론, 환각 인용 필터, 트랜스크립트 생성."""

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

# nodes/__init__ 이 동명 함수를 재수출하므로 모듈 객체를 직접 가져온다.
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


def test_no_evidence_runs_commonsense_debate(monkeypatch):
    """증거 0건(코퍼스 밖 주장)도 토론을 생략하지 않고 일반 상식 토론을 진행한다."""
    prompts_seen: list[str] = []

    def fake_invoke(prompt, schema, *, default):
        prompts_seen.append(prompt)
        return SideArgument(
            summary=f"상식 논거{len(prompts_seen)}",
            cited_snippet_ids=["ev_fake"],  # 빈 풀 → 사후 필터로 제거돼야 함
        )

    monkeypatch.setattr(deb, "structured_invoke", fake_invoke)
    update = deb.adversarial_debate(
        {"claims": [_claim()], "evidence_pool": [], "loop_count": 0}
    )
    assert len(prompts_seen) == 3  # 검사 → 변호 → 검사 재반박 그대로
    assert all("(제공된 증거가 없습니다)" in p for p in prompts_seen)

    [pair] = update["arguments"]
    assert pair.prosecution.cited_snippet_ids == []
    assert pair.defense.cited_snippet_ids == []

    turns = update["debate_transcript"]
    assert turns[0].role == "판사"  # 상식 토론 안내 시스템 턴
    assert "일반 상식" in turns[0].text
    assert [t.role for t in turns[1:]] == ["검사", "변호", "검사"]


def test_one_empty_side_does_not_trigger_commonsense(monkeypatch):
    """주장에 증거가 있는데 한쪽 풀만 비면 상식 모드를 발동하지 않는다."""
    prompts_seen: list[str] = []

    def fake_invoke(prompt, schema, *, default):
        prompts_seen.append(prompt)
        return SideArgument(summary="논거", cited_snippet_ids=[])

    monkeypatch.setattr(deb, "structured_invoke", fake_invoke)
    pool = [_ev("ev_pro", STANCE_PROSECUTION)]  # 변호 측 풀만 빈 상황
    update = deb.adversarial_debate(
        {"claims": [_claim()], "evidence_pool": pool, "loop_count": 0}
    )
    # 프롬프트 규칙 문구에도 트리거 문자열이 들어 있으므로, 실제 증거
    # 블록([제공된 증거] 헤더 뒤)만 떼어 검사한다.
    defense_block = prompts_seen[1].split("[제공된 증거]\n")[-1]
    assert "(제공된 증거가 없습니다)" not in defense_block
    assert "귀측 리서처가 회수한 증거가 없습니다" in defense_block
    assert all(t.role != "판사" for t in update["debate_transcript"])


def test_retrieval_failure_notice_differs(monkeypatch):
    """검색 인프라 장애는 범위 밖 주장 안내문과 구분되어야 한다."""
    monkeypatch.setattr(
        deb, "structured_invoke",
        lambda prompt, schema, *, default: SideArgument(
            summary="논거", cited_snippet_ids=[]
        ),
    )
    update = deb.adversarial_debate(
        {
            "claims": [_claim()], "evidence_pool": [],
            "loop_count": 0, "retrieval_failed": True,
        }
    )
    notice = update["debate_transcript"][0]
    assert notice.role == "판사"
    assert "일시적으로 실패" in notice.text
