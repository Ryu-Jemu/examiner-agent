"""LangGraph State 스키마.

리듀서(`Annotated[list, add]`)와 단일-쓰기 필드의 구분이 그래프 정확성의 핵심이다.

- 같은 superstep 에서 두 노드가 *리듀서 없는* 같은 키에 쓰면 LangGraph 가
  `InvalidUpdateError` 를 던진다. `tag_techniques` 는 메인 체인과 병렬로 도므로
  병렬 분기가 건드리는 리스트는 반드시 `add` 리듀서를 가진다.
- `loop_count` 는 judge 만 쓰는 단일-쓰기 카운터이므로 리듀서를 두지 않는다
  (`add` 를 두면 값이 합산되어 이중 카운트된다).
"""

from __future__ import annotations

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
    # --- 입력 (START 에서 1회 설정, 불변) ---
    input_text: str

    # --- 노드 산출물 (주석: 쓰는 노드 / 리듀서 여부) ---
    claims: list[Claim]  # extract_claims, 단일 쓰기 → 교체
    evidence_pool: Annotated[list[EvidenceItem], add]  # retrieve, 루프 누적
    prev_pool_size: int  # 직전 풀 크기(루프 종료 판정용); retrieve 가 갱신
    arguments: Annotated[list[ArgumentPair], add]  # debate, 라운드 누적
    verdicts: list[Verdict]  # judge, 최신 판정으로 교체
    refutation_log: Annotated[list[RefutationEntry], add]  # self-refute 누적
    technique_tags: Annotated[list[TechniqueTag], add]  # 병렬 → 리듀서 필수

    # --- 루프 제어 (judge 단일 쓰기) ---
    loop_count: int  # 카운터, 합산 금지 → 교체
    last_confidence: float  # 신뢰도 델타 종료 판정용
    route_decision: str  # 다음 분기: retrieve_evidence | synthesize

    # --- 종단 ---
    final_report: Optional[FinalReport]
