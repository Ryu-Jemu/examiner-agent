"""최종 리포트 합성 노드. 핵심 수치는 결정론적 계산, 반론 카드만 LLM 이 작성한다."""

from .. import prompts
from ..llm import structured_invoke
from ..models import (
    ArgumentPair,
    ClaimBreakdown,
    Claim,
    EvidenceItem,
    FinalReport,
    RebuttalCard,
    TechniqueTag,
    Verdict,
    VerdictLabel,
)
from ..state import FactCheckState

# 등급 → 진실성 점수(결정론적 종합용)
_TRUTH_SCORE = {
    VerdictLabel.TRUE: 1.0,
    VerdictLabel.MOSTLY_TRUE: 0.75,
    VerdictLabel.INSUFFICIENT: 0.5,
    VerdictLabel.MOSTLY_FALSE: 0.25,
    VerdictLabel.FALSE: 0.0,
}


def _score_to_label(score: float) -> VerdictLabel:
    if score >= 0.875:
        return VerdictLabel.TRUE
    if score >= 0.625:
        return VerdictLabel.MOSTLY_TRUE
    if score >= 0.375:
        return VerdictLabel.INSUFFICIENT
    if score >= 0.125:
        return VerdictLabel.MOSTLY_FALSE
    return VerdictLabel.FALSE


def _source_label(ev: EvidenceItem) -> str:
    return f"{ev.source} ({ev.source_type.value})"


def _fallback_rebuttal(overall: VerdictLabel) -> str:
    """LLM 반론 생성 실패 시 등급에 일관된 안전 문구."""
    if overall in (VerdictLabel.FALSE, VerdictLabel.MOSTLY_FALSE):
        return (
            f"확인 결과 이 내용은 사실과 다르거나 오해의 소지가 큰 것으로 보입니다"
            f"('{overall.value}'). 공유하시기 전에 공신력 있는 출처를 함께 확인해"
            " 보시면 좋겠습니다."
        )
    if overall in (VerdictLabel.TRUE, VerdictLabel.MOSTLY_TRUE):
        return (
            f"확인 결과 이 내용은 대체로 사실에 부합하는 것으로 보입니다"
            f"('{overall.value}'). 다만 출처를 함께 확인하면 더 정확합니다."
        )
    return (
        "현재 확보한 근거만으로는 사실 여부를 단정하기 어렵습니다"
        f"('{overall.value}'). 공신력 있는 출처를 함께 확인해 보시길 권합니다."
    )


def _build_breakdown(
    claims: list[Claim],
    verdicts: list[Verdict],
    arguments: list[ArgumentPair],
    pool: list[EvidenceItem],
) -> list[ClaimBreakdown]:
    claim_text = {c.claim_id: c.text for c in claims}
    by_id = {e.snippet_id: e for e in pool}

    # claim_id → 최신 라운드의 논거(지지=변호 인용, 반박=검사 인용)
    latest_args: dict[int, ArgumentPair] = {}
    for ap in arguments:
        cur = latest_args.get(ap.claim_id)
        if cur is None or ap.loop >= cur.loop:
            latest_args[ap.claim_id] = ap

    breakdowns = []
    for v in verdicts:
        ap = latest_args.get(v.claim_id)
        supporting, refuting = [], []
        if ap:
            supporting = [
                _source_label(by_id[s])
                for s in ap.defense.cited_snippet_ids
                if s in by_id
            ]
            refuting = [
                _source_label(by_id[s])
                for s in ap.prosecution.cited_snippet_ids
                if s in by_id
            ]
        breakdowns.append(
            ClaimBreakdown(
                claim_id=v.claim_id,
                text=claim_text.get(v.claim_id, ""),
                label=v.label,
                confidence=v.confidence,
                supporting_sources=sorted(set(supporting)),
                refuting_sources=sorted(set(refuting)),
                self_refutation=v.self_refutation,
            )
        )
    return breakdowns


def _format_report_block(
    overall: VerdictLabel,
    breakdowns: list[ClaimBreakdown],
    tags: list[TechniqueTag],
) -> str:
    lines = [f"종합 등급: {overall.value}"]
    for b in breakdowns:
        sup = ", ".join(b.supporting_sources) or "없음"
        ref = ", ".join(b.refuting_sources) or "없음"
        lines.append(
            f"- 주장: {b.text}\n  판정: {b.label.value} (신뢰도 {b.confidence:.2f})"
            f"\n  지지 출처: {sup}\n  반박 출처: {ref}"
        )
    if tags:
        tag_names = ", ".join(sorted({t.tag.value for t in tags}))
        lines.append(f"발견된 조작 기법: {tag_names}")
    return "\n".join(lines)


def synthesize(state: FactCheckState) -> dict:
    claims: list[Claim] = state.get("claims", []) or []
    verdicts: list[Verdict] = state.get("verdicts", []) or []
    arguments: list[ArgumentPair] = state.get("arguments", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    tags: list[TechniqueTag] = state.get("technique_tags", []) or []

    breakdowns = _build_breakdown(claims, verdicts, arguments, pool)

    if verdicts:
        avg_truth = (
            sum(_TRUTH_SCORE[v.label] for v in verdicts) / len(verdicts)
        )
        overall = _score_to_label(avg_truth)
        overall_conf = sum(v.confidence for v in verdicts) / len(verdicts)
    else:
        overall = VerdictLabel.INSUFFICIENT
        overall_conf = 0.0

    report_block = _format_report_block(overall, breakdowns, tags)

    # 반론 카드(LLM) — 실패해도 안전하고 등급에 일관된 기본 문구
    fallback = RebuttalCard(rebuttal_card=_fallback_rebuttal(overall))
    rebuttal = structured_invoke(
        prompts.render("rebuttal", report_block=report_block),
        RebuttalCard,
        default=fallback,
    )

    report = FinalReport(
        overall_grade=overall,
        overall_confidence=overall_conf,
        claim_breakdown=breakdowns,
        technique_tags=tags,
        rebuttal_card=rebuttal.rebuttal_card,
    )
    return {"final_report": report}
