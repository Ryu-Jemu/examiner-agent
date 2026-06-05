"""그래프 토폴로지 경험적 검증.

가장 위험한 부분: tag_techniques(병렬) → synthesize 의 fan-in 조인이 judge 루프와
함께 올바르게 동작하는가? (조기 발화/데드락 없이, synthesize 가 두 분기를 모두 받은
뒤 정확히 한 번 실행되는가?)

LLM 을 호출하지 않도록 노드 본문을 스텁으로 교체해 실제 build_graph 배선만 검증한다.
"""

from __future__ import annotations

from factchecker.models import (
    Claim,
    ClaimType,
    EvidenceItem,
    FinalReport,
    TechniqueTag,
    TechniqueTagName,
    Verdict,
    VerdictLabel,
)


def test_parallel_join_with_loop(monkeypatch):
    from factchecker import graph as gmod

    retrieve_calls = []
    synth_states = []

    def stub_extract(state):
        return {
            "claims": [Claim(claim_id=0, text="주장", claim_type=ClaimType.FACT, checkable=True)],
            "loop_count": 0,
            "last_confidence": 0.0,
            "prev_pool_size": 0,
        }

    def stub_retrieve(state):
        prev = len(state.get("evidence_pool", []))
        retrieve_calls.append(prev)
        new = EvidenceItem(
            claim_id=0, snippet_id=f"ev{prev}", snippet="근거", source="s"
        )
        return {"evidence_pool": [new], "prev_pool_size": prev}

    def stub_debate(state):
        return {"arguments": []}

    def stub_judge(state):
        loop = state.get("loop_count", 0)
        new_loop = loop + 1
        decision = "retrieve_evidence" if new_loop < 2 else "synthesize"
        return {
            "verdicts": [Verdict(claim_id=0, label=VerdictLabel.FALSE, confidence=0.9)],
            "loop_count": new_loop,
            "last_confidence": 0.9,
            "route_decision": decision,
        }

    def stub_tag(state):
        return {
            "technique_tags": [
                TechniqueTag(tag=TechniqueTagName.EMOTION, evidence_sentence="충격!")
            ]
        }

    def stub_synth(state):
        synth_states.append(dict(state))
        return {
            "final_report": FinalReport(
                overall_grade=VerdictLabel.FALSE, overall_confidence=0.9
            )
        }

    monkeypatch.setattr(gmod, "extract_claims", stub_extract)
    monkeypatch.setattr(gmod, "retrieve_evidence", stub_retrieve)
    monkeypatch.setattr(gmod, "adversarial_debate", stub_debate)
    monkeypatch.setattr(gmod, "judge", stub_judge)
    monkeypatch.setattr(gmod, "tag_techniques", stub_tag)
    monkeypatch.setattr(gmod, "synthesize", stub_synth)

    compiled = gmod.build_graph().compile()
    final = compiled.invoke({"input_text": "충격! 큰일"}, config={"recursion_limit": 50})

    # synthesize 가 정확히 한 번 실행되어야 함(조기 발화/중복 없음)
    assert len(synth_states) == 1
    # 병렬 분기 결과(technique_tags)가 조인 시점에 존재해야 함
    assert synth_states[0].get("technique_tags"), "병렬 분기가 합류하지 않음"
    # 루프가 한 번 돌아 retrieve 가 2회 호출되어야 함
    assert len(retrieve_calls) == 2
    # 최종 리포트 생성 확인
    assert final["final_report"].overall_grade == VerdictLabel.FALSE
