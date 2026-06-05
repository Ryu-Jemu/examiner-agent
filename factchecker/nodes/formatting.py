"""프롬프트용 블록 포매팅 헬퍼."""

from __future__ import annotations

from ..models import ArgumentPair, Claim, EvidenceItem, Verdict


def format_evidence_block(items: list[EvidenceItem]) -> str:
    if not items:
        return "(제공된 증거가 없습니다)"
    lines = []
    for it in items:
        cred = f"{it.credibility:.2f}"
        lines.append(
            f"- id={it.snippet_id} | 출처={it.source} "
            f"({it.source_type.value}, 신뢰도 {cred})\n  내용: {it.snippet}"
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
        lines.append(
            f"[claim_id={ap.claim_id} | loop={ap.loop}]\n"
            f"  검사: {ap.prosecution.summary} (인용: {pro_ids})\n"
            f"  변호: {ap.defense.summary} (인용: {def_ids})"
        )
    return "\n".join(lines)


def format_verdicts_block(verdicts: list[Verdict]) -> str:
    if not verdicts:
        return "(판정이 없습니다)"
    lines = []
    for v in verdicts:
        chain = ", ".join(v.evidence_chain) or "없음"
        lines.append(
            f"- claim_id={v.claim_id} | 판정={v.label.value} "
            f"| 신뢰도={v.confidence:.2f} | 근거사슬=[{chain}]\n  근거: {v.rationale}"
        )
    return "\n".join(lines)


def evidence_for_claim(pool: list[EvidenceItem], claim_id: int) -> list[EvidenceItem]:
    return [e for e in pool if e.claim_id == claim_id]
