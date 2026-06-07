"""구조화 출력·State 하위객체용 Pydantic 스키마. enum 은 Gemini 호환을 위해 str Enum."""

from enum import Enum

from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    FACT = "사실주장"
    OPINION = "의견"


class SourceType(str, Enum):
    GOV = "정부·공공"
    ACADEMIC = "학술"
    NEWS = "뉴스"
    ORG = "기관"
    BLOG = "블로그·SNS"
    UNKNOWN = "출처불명"


class VerdictLabel(str, Enum):
    TRUE = "사실"
    MOSTLY_TRUE = "대체로 사실"
    INSUFFICIENT = "불충분(판단 불가)"
    MOSTLY_FALSE = "대체로 거짓"
    FALSE = "거짓·오도"


class TechniqueTagName(str, Enum):
    EMOTION = "감정 자극"
    FAKE_AUTHORITY = "가짜 전문가·권위"
    CHERRY_PICKING = "가짜 통계·체리피킹"
    FALSE_DICHOTOMY = "거짓 이분법"


# 출처 유형 → 신뢰도(결정론적 매핑; retrieve_evidence 에서 사용)
SOURCE_CREDIBILITY: dict[SourceType, float] = {
    SourceType.GOV: 0.90,
    SourceType.ACADEMIC: 0.85,
    SourceType.NEWS: 0.70,
    SourceType.ORG: 0.65,
    SourceType.BLOG: 0.40,
    SourceType.UNKNOWN: 0.20,
}


class Claim(BaseModel):
    claim_id: int = Field(description="0부터 시작하는 주장 식별자")
    text: str = Field(description="원문에서 추출한 검증 대상 문장")
    claim_type: ClaimType = Field(description="사실주장 또는 의견")
    checkable: bool = Field(description="검증 가능한 사실주장이면 true")


class ClaimList(BaseModel):
    """extract_claims 노드의 구조화 출력."""

    claims: list[Claim] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    claim_id: int
    snippet_id: str
    snippet: str
    source: str
    source_type: SourceType = SourceType.UNKNOWN
    credibility: float = Field(ge=0.0, le=1.0, default=0.2)
    url: str | None = None
    stance: str | None = None


class SideArgument(BaseModel):
    """검사 또는 변호 한 측의 논거."""

    summary: str = Field(description="이 측의 핵심 주장 요약")
    cited_snippet_ids: list[str] = Field(
        default_factory=list,
        description="반드시 제공된 증거 스니펫의 id 중에서만 인용",
    )


class ArgumentPair(BaseModel):
    claim_id: int
    loop: int
    prosecution: SideArgument  # 검사 (거짓·오도 입증)
    defense: SideArgument      # 변호 (참 입증)


class Verdict(BaseModel):
    claim_id: int
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chain: list[str] = Field(
        default_factory=list, description="결론 근거가 된 snippet_id 목록(순서대로)"
    )
    rationale: str = Field(default="", description="판정 근거 설명")
    # 자가 반박을 판정과 같은 호출에서 함께 산출 → LLM 호출 1회 절약
    self_refutation: str = Field(
        default="", description="이 판정을 뒤집을 수 있는 가장 강한 반론(레드팀)"
    )
    survives_refutation: bool = Field(
        default=True, description="위 반론에도 판정이 유지되면 true, 흔들리면 false"
    )
    needs_more_evidence: bool = Field(
        default=False, description="증거 부족으로 추가 검색이 필요하면 true"
    )


class VerdictList(BaseModel):
    """judge 노드의 판정 구조화 출력."""

    verdicts: list[Verdict] = Field(default_factory=list)


class RefutationEntry(BaseModel):
    """자가 반박 기록(judge 가 각 판정에서 구성)."""

    loop: int
    claim_id: int
    challenge: str
    survived: bool


class TechniqueTag(BaseModel):
    tag: TechniqueTagName
    evidence_sentence: str = Field(description="해당 기법이 드러난 본문 문장")
    library_entry_id: str = Field(default="", description="매칭된 기법 라이브러리 id")
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class TechniqueTagList(BaseModel):
    """tag_techniques 노드의 구조화 출력."""

    technique_tags: list[TechniqueTag] = Field(default_factory=list)


class ClaimBreakdown(BaseModel):
    claim_id: int
    text: str
    label: VerdictLabel
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: list[str] = Field(default_factory=list)
    refuting_sources: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    overall_grade: VerdictLabel
    overall_confidence: float = Field(ge=0.0, le=1.0)
    claim_breakdown: list[ClaimBreakdown] = Field(default_factory=list)
    technique_tags: list[TechniqueTag] = Field(default_factory=list)
    rebuttal_card: str = ""


class RebuttalCard(BaseModel):
    """synthesize 노드에서 LLM 이 작성하는 반론 카드 스키마."""

    rebuttal_card: str = Field(description="단톡방에 바로 붙여넣을 짧고 차분한 한국어 메시지")
