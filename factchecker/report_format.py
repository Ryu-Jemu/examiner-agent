"""FinalReport 를 사람이 읽기 좋은 한국어 텍스트/마크다운으로 렌더링.

CLI 출력과 Gradio 표시에 공통으로 사용한다.
"""

from __future__ import annotations

from .models import FinalReport


def report_to_text(report: FinalReport) -> str:
    """CLI 용 플레인 텍스트."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"📋 종합 신뢰 등급: {report.overall_grade.value} "
                 f"(신뢰도 {report.overall_confidence:.0%})")
    lines.append("=" * 60)

    if report.claim_breakdown:
        lines.append("\n[주장별 판정]")
        for b in report.claim_breakdown:
            lines.append(f"  • {b.text}")
            lines.append(f"     판정: {b.label.value} (신뢰도 {b.confidence:.0%})")
            if b.supporting_sources:
                lines.append(f"     지지 출처: {', '.join(b.supporting_sources)}")
            if b.refuting_sources:
                lines.append(f"     반박 출처: {', '.join(b.refuting_sources)}")
    else:
        lines.append("\n[주장별 판정] 검증 가능한 사실주장을 찾지 못했습니다.")

    if report.technique_tags:
        lines.append("\n[발견된 조작 기법]")
        seen = set()
        for t in report.technique_tags:
            if t.tag.value in seen:
                continue
            seen.add(t.tag.value)
            lines.append(f"  • {t.tag.value}: \"{t.evidence_sentence}\"")
    else:
        lines.append("\n[발견된 조작 기법] 뚜렷한 조작 기법이 식별되지 않았습니다.")

    lines.append("\n[💬 반론 카드 — 단톡방에 붙여넣기]")
    lines.append(report.rebuttal_card)
    lines.append("=" * 60)
    return "\n".join(lines)


def report_to_markdown(report: FinalReport) -> str:
    """Gradio 표시용 마크다운."""
    md = []
    md.append(f"## 📋 종합 신뢰 등급: **{report.overall_grade.value}**")
    md.append(f"신뢰도: **{report.overall_confidence:.0%}**\n")

    md.append("### 주장별 판정")
    if report.claim_breakdown:
        for b in report.claim_breakdown:
            md.append(f"- **{b.text}**")
            md.append(f"  - 판정: `{b.label.value}` (신뢰도 {b.confidence:.0%})")
            if b.supporting_sources:
                md.append(f"  - 지지 출처: {', '.join(b.supporting_sources)}")
            if b.refuting_sources:
                md.append(f"  - 반박 출처: {', '.join(b.refuting_sources)}")
    else:
        md.append("- 검증 가능한 사실주장을 찾지 못했습니다.")

    md.append("\n### 🚩 발견된 조작 기법")
    if report.technique_tags:
        seen = set()
        for t in report.technique_tags:
            if t.tag.value in seen:
                continue
            seen.add(t.tag.value)
            md.append(f"- **{t.tag.value}**: \"{t.evidence_sentence}\"")
    else:
        md.append("- 뚜렷한 조작 기법이 식별되지 않았습니다.")

    md.append("\n### 💬 반론 카드 (단톡방에 붙여넣기)")
    md.append(f"> {report.rebuttal_card}")
    return "\n".join(md)
