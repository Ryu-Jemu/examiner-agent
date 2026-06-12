"""LangGraph State 스키마."""

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
    input_text: str  # START에서 1회 설정
    input_image: Optional[str]  # 첨부 이미지(data URL), 없으면 미설정

    claims: list[Claim]  # extract_claims, 단일 쓰기로 교체
    evidence_pool: Annotated[list[EvidenceItem], add]  # retrieve, 루프 누적
    prev_pool_size: int  # 직전 풀 크기(루프 종료 판정용)
    # 벡터스토어 로드/회수 실패 여부. 코퍼스에 증거 없음(범위 밖)과 구분해 인프라 장애를 범위 밖 주장으로 위장하지 않게 한다.
    retrieval_failed: bool
    arguments: Annotated[list[ArgumentPair], add]  # debate, 라운드 누적
    # 법정 대화 기록(검사·변호 3턴 + 판사 시스템 턴). 라운드마다 누적되어야 하므로 add 리듀서 필수.
    debate_transcript: Annotated[list[DebateTurn], add]
    verdicts: list[Verdict]  # judge, 최신 판정으로 교체
    refutation_log: Annotated[list[RefutationEntry], add]  # self-refute 누적
    technique_tags: Annotated[list[TechniqueTag], add]  # 병렬 분기, 리듀서 필수

    loop_count: int  # judge 단일 쓰기 카운터(합산 금지)
    last_confidence: float
    route_decision: str  # retrieve_evidence | synthesize

    final_report: Optional[FinalReport]
