"""LangGraph State 스키마. 병렬 분기가 쓰는 리스트는 add 리듀서, loop_count 는 단일 쓰기."""

from operator import add
from typing import Annotated, Optional

from typing_extensions import TypedDict

from .models import (
    ArgumentPair,
    Claim,
    EvidenceItem,
    FinalReport,
    RefutationEntry,
    TechniqueTag,
    Verdict,
)


class FactCheckState(TypedDict, total=False):
    input_text: str  # START 에서 1회 설정

    claims: list[Claim]  # extract_claims, 단일 쓰기 → 교체
    evidence_pool: Annotated[list[EvidenceItem], add]  # retrieve, 루프 누적
    prev_pool_size: int  # 직전 풀 크기(루프 종료 판정용)
    arguments: Annotated[list[ArgumentPair], add]  # debate, 라운드 누적
    verdicts: list[Verdict]  # judge, 최신 판정으로 교체
    refutation_log: Annotated[list[RefutationEntry], add]  # self-refute 누적
    technique_tags: Annotated[list[TechniqueTag], add]  # 병렬 → 리듀서 필수

    loop_count: int  # judge 단일 쓰기 카운터(합산 금지)
    last_confidence: float
    route_decision: str  # retrieve_evidence | synthesize

    final_report: Optional[FinalReport]
