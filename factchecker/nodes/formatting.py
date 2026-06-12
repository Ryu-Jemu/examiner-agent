"""프롬프트용 블록 포매팅 헬퍼."""

from ..models import STANCE_BOTH, ArgumentPair, Claim, EvidenceItem


def format_evidence_block(items: list[EvidenceItem]) -> str:
    if not items:
        return "(제공된 증거가 없습니다)"
    lines = []
    for it in items:
        cred = f"{it.credibility:.2f}"
        # 매칭도와 작성 시점을 노출해 판사·토론자가 직접 가늠하게 함
        rel = f", 매칭도 {it.relevance:.2f}" if it.relevance is not None else ""
        date = f", 작성시점 {it.date}" if it.date else ""
        lines.append(
            f"- id={it.snippet_id} | 출처={it.source} "
            f"({it.source_type.value}, 신뢰도 {cred}{rel}{date})\n"
            f"  내용: {it.snippet}"
        )
    return "\n".join(lines)


def format_claims_block(claims: list[Claim]) -> str:
    checkable = [c for c in claims if c.checkable]
    if not checkable:
        return "(검증 대상 사실주장이 없습니다)"
    return "\n".join(f"- claim_id={c.claim_id}: {c.text}" for c in checkable)


def format_arguments_block(arguments: list[ArgumentPair]) -> str:
    if not arguments:
        return "(아직 논거가 없습니다)"
    lines = []
    for ap in arguments:
        pro_ids = ", ".join(ap.prosecution.cited_snippet_ids) or "없음"
        def_ids = ", ".join(ap.defense.cited_snippet_ids) or "없음"
        block = (
            f"[claim_id={ap.claim_id} | loop={ap.loop}]\n"
            f"  검사: {ap.prosecution.summary} (인용: {pro_ids})\n"
            f"  변호: {ap.defense.summary} (인용: {def_ids})"
        )
        if ap.prosecution_rebuttal is not None:
            reb_ids = (
                ", ".join(ap.prosecution_rebuttal.cited_snippet_ids) or "없음"
            )
            block += (
                f"\n  검사 재반박: {ap.prosecution_rebuttal.summary}"
                f" (인용: {reb_ids})"
            )
        lines.append(block)
    return "\n".join(lines)


def evidence_for_claim(
    pool: list[EvidenceItem], claim_id: int
) -> list[EvidenceItem]:
    return [e for e in pool if e.claim_id == claim_id]


def evidence_for_side(
    pool: list[EvidenceItem], claim_id: int, side: str
) -> list[EvidenceItem]:
    """해당 측 리서처 전속 풀: 자기 스탠스 + 양측 공통(both) + 미태깅(하위호환)."""
    return [
        e
        for e in pool
        if e.claim_id == claim_id
        and (e.stance in (side, STANCE_BOTH) or e.stance is None)
    ]
