"""models 스키마 테스트: enum, 범위 검증, 신뢰도 매핑."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_credibility_mapping_covers_all_source_types():
    from factchecker.models import SOURCE_CREDIBILITY, SourceType

    for st in SourceType:
        assert st in SOURCE_CREDIBILITY
        assert 0.0 <= SOURCE_CREDIBILITY[st] <= 1.0


def test_verdict_confidence_bounds():
    from factchecker.models import Verdict, VerdictLabel

    with pytest.raises(ValidationError):
        Verdict(claim_id=0, label=VerdictLabel.TRUE, confidence=1.5)

    v = Verdict(claim_id=0, label=VerdictLabel.INSUFFICIENT, confidence=0.0)
    assert v.label.value == "불충분(판단 불가)"


def test_evidence_item_defaults():
    from factchecker.models import EvidenceItem, SourceType

    e = EvidenceItem(claim_id=1, snippet_id="x", snippet="t", source="s")
    assert e.source_type == SourceType.UNKNOWN
    assert 0.0 <= e.credibility <= 1.0


def test_five_grade_labels():
    from factchecker.models import VerdictLabel

    values = {v.value for v in VerdictLabel}
    assert values == {
        "사실",
        "대체로 사실",
        "불충분(판단 불가)",
        "대체로 거짓",
        "거짓·오도",
    }


def test_four_techniques():
    from factchecker.models import TechniqueTagName

    values = {t.value for t in TechniqueTagName}
    assert values == {
        "감정 자극",
        "가짜 전문가·권위",
        "가짜 통계·체리피킹",
        "거짓 이분법",
    }
