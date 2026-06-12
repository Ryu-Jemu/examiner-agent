"""검사·변호 3턴 순차 토론 노드. 인용 id 사후 필터링으로 환각 가드. 코퍼스 증거 0건 주장은 일반 상식 토론으로 전환."""

import logging

from .. import prompts
from ..llm import structured_invoke
from ..models import (
    STANCE_DEFENSE,
    STANCE_PROSECUTION,
    ArgumentPair,
    Claim,
    DebateTurn,
    EvidenceItem,
    SideArgument,
)
from ..state import FactCheckState
from .formatting import (
    evidence_for_claim,
    evidence_for_side,
    format_evidence_block,
)

logger = logging.getLogger("factchecker.nodes.debate")

# 한쪽 리서처 풀만 빈 경우의 안내문(상식 토론 트리거 문자열과 달라야 함).
_EMPTY_SIDE_BLOCK = (
    "(귀측 리서처가 회수한 증거가 없습니다. 상대측 증거만 존재합니다. "
    "인용 없이 단정하지 말고, 증거가 부족하다는 사실을 summary 에 솔직히 "
    "적으세요.)"
)


def _filter_citations(side: SideArgument, valid_ids: set[str]) -> SideArgument:
    """지어낸 인용 id 를 제거한다(환각 가드)."""
    kept, dropped = [], []
    for sid in side.cited_snippet_ids:
        (kept if sid in valid_ids else dropped).append(sid)
    if dropped:
        logger.warning("환각 인용 제거: %s (유효 id 아님)", dropped)
    return SideArgument(summary=side.summary, cited_snippet_ids=kept)


def _side_block(side: SideArgument) -> str:
    ids = ", ".join(side.cited_snippet_ids) or "없음"
    return f"{side.summary} (인용: {ids})"


def adversarial_debate(state: FactCheckState) -> dict:
    claims: list[Claim] = state.get("claims", []) or []
    pool: list[EvidenceItem] = state.get("evidence_pool", []) or []
    loop = state.get("loop_count", 0)

    pairs: list[ArgumentPair] = []
    transcript: list[DebateTurn] = []
    for claim in claims:
        if not claim.checkable:
            continue
        ev_all = evidence_for_claim(pool, claim.claim_id)
        # 코퍼스 증거 0건(범위 밖 주장): 일반 상식 토론으로 전환. 환각 가드는 프롬프트 날조 금지 규칙, 인용 id 사후 필터, 판사 단계 신뢰도 상한 3중.
        if not ev_all:
            if state.get("retrieval_failed", False):
                # 인프라 장애를 '범위 밖 주장'으로 위장하지 않도록 안내문 분기.
                notice = (
                    f"「{claim.text}」 증거 검색이 일시적으로 실패하여 "
                    "코퍼스 증거 없이 진행합니다(주제가 범위 밖이 아닐 수 "
                    "있음). 양측이 일반 상식 수준에서 토론합니다(인용 없음)."
                )
            else:
                notice = (
                    f"「{claim.text}」 이 주장과 관련된 코퍼스 증거가 "
                    "없어, 양측이 일반 상식 수준에서 토론합니다"
                    "(증거 인용 없음)."
                )
            transcript.append(
                DebateTurn(
                    claim_id=claim.claim_id, loop=loop, turn=99, role="판사",
                    text=notice,
                )
            )

        # 양측 전속 리서처 풀(자기 스탠스 + both + 미태깅)
        ev_pro = evidence_for_side(pool, claim.claim_id, STANCE_PROSECUTION)
        ev_def = evidence_for_side(pool, claim.claim_id, STANCE_DEFENSE)
        pro_ids = {e.snippet_id for e in ev_pro}
        def_ids = {e.snippet_id for e in ev_def}
        # 한쪽 풀만 비면 별도 안내문으로 '증거 부족 솔직 진술' 경로.
        pro_block = (
            format_evidence_block(ev_pro)
            if (ev_pro or not ev_all) else _EMPTY_SIDE_BLOCK
        )
        def_block = (
            format_evidence_block(ev_def)
            if (ev_def or not ev_all) else _EMPTY_SIDE_BLOCK
        )
        empty = SideArgument(summary="(논거 생성 실패)", cited_snippet_ids=[])

        # 검사 발언
        prosecution = structured_invoke(
            prompts.render(
                "prosecution",
                claim_text=claim.text,
                evidence_block=pro_block,
            ),
            SideArgument,
            default=empty,
        )
        prosecution = _filter_citations(prosecution, pro_ids)

        # 변호 반박(검사 논거를 입력으로 받음)
        defense = structured_invoke(
            prompts.render(
                "defense",
                claim_text=claim.text,
                opponent_block=_side_block(prosecution),
                evidence_block=def_block,
            ),
            SideArgument,
            default=empty,
        )
        defense = _filter_citations(defense, def_ids)

        # 검사 재반박(변호 반박을 입력으로 받음)
        rebuttal = structured_invoke(
            prompts.render(
                "prosecution_rebuttal",
                claim_text=claim.text,
                own_block=_side_block(prosecution),
                opponent_block=_side_block(defense),
                evidence_block=pro_block,
            ),
            SideArgument,
            default=empty,
        )
        rebuttal = _filter_citations(rebuttal, pro_ids)

        pairs.append(
            ArgumentPair(
                claim_id=claim.claim_id,
                loop=loop,
                prosecution=prosecution,
                defense=defense,
                prosecution_rebuttal=rebuttal,
            )
        )
        for turn, (role, side) in enumerate(
            (("검사", prosecution), ("변호", defense), ("검사", rebuttal))
        ):
            transcript.append(
                DebateTurn(
                    claim_id=claim.claim_id,
                    loop=loop,
                    turn=turn,
                    role=role,
                    text=side.summary,
                    cited_snippet_ids=side.cited_snippet_ids,
                )
            )

    # arguments / debate_transcript 모두 add 리듀서로 라운드 누적
    return {"arguments": pairs, "debate_transcript": transcript}
