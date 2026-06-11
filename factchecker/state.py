"""LangGraph State 스키마. 병렬 분기가 쓰는 리스트는 add 리듀서, loop_count 는 단일 쓰기."""

from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from .models import (
    ArgumentPair,
    Claim,
    DebateTurn,
    EvidenceItem,
    FinalReport,
    RefutationEntry,
    TechniqueTag,
    Verdict,
)


class FactCheckState(TypedDict, total=False):
    input_text: str  # START 에서 1회 설정
    input_image: Optional[str]  # 첨부 이미지(data URL), 없으면 미설정

    claims: list[Claim]  # extract_claims, 단일 쓰기 → 교체
    evidence_pool: Annotated[list[EvidenceItem], add]  # retrieve, 루프 누적
    prev_pool_size: int  # 직전 풀 크기(루프 종료 판정용)
    arguments: Annotated[list[ArgumentPair], add]  # debate, 라운드 누적
    # 법정 대화 기록(검사·변호 3턴 + 판사 시스템 턴). 루프 라운드마다 누적
    # 되어야 하므로 add 리듀서 필수(없으면 2라운드에서 1라운드 기록 소실).
    debate_transcript: Annotated[list[DebateTurn], add]
    verdicts: list[Verdict]  # judge, 최신 판정으로 교체
    refutation_log: Annotated[list[RefutationEntry], add]  # self-refute 누적
    technique_tags: Annotated[list[TechniqueTag], add]  # 병렬 → 리듀서 필수

    loop_count: int  # judge 단일 쓰기 카운터(합산 금지)
    last_confidence: float
    route_decision: str  # retrieve_evidence | synthesize

    final_report: Optional[FinalReport]
